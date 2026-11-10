"""Neumann/zero-flux neural primitives for the audited VCH generator geometry.

The prepared VCH data were generated with ghost-point Neumann boundaries, not
a periodic torus.  ``replicate`` padding is used here as a cell-centred
zero-normal-flux stencil: a boundary face has no neighbour beyond the domain.
The conservative helper is written directly as face fluxes so its spatial sum
is zero (up to floating-point roundoff) without wrapping opposite edges.
"""

from __future__ import annotations

import torch
from torch import nn


def trapezoidal_node_weights(value: torch.Tensor) -> torch.Tensor:
    """Return unnormalised 2-D trapezoidal quadrature weights on a node grid.

    The VCH-v3.1 generator stores values at 128 *nodes* (127 intervals), and
    its mass audit uses tensor-product trapezoidal weights.  A uniform pixel
    mean is therefore not the discrete invariant of the source solver: edge
    nodes carry half weight and corner nodes one quarter weight.
    """

    if value.ndim < 2 or value.shape[-2] < 2 or value.shape[-1] < 2:
        raise ValueError("trapezoidal_node_weights requires at least a 2x2 spatial grid.")
    weights = torch.ones_like(value)
    weights[..., 0, :] *= 0.5
    weights[..., -1, :] *= 0.5
    weights[..., :, 0] *= 0.5
    weights[..., :, -1] *= 0.5
    return weights


def trapezoidal_mass(value: torch.Tensor) -> torch.Tensor:
    """Compute the source-solver-compatible mass, omitting constant dx*dy."""

    return (value * trapezoidal_node_weights(value)).sum(dim=(-2, -1), keepdim=True)


def project_trapezoidal_mass(
    value: torch.Tensor,
    reference_mass: torch.Tensor,
    *,
    lower: float = -0.99,
    upper: float = 0.99,
    interior_margin: float = 0.005,
) -> torch.Tensor:
    """Bound and project a field to the reference trapezoidal mass.

    This follows the source solver's principle: correct only unsaturated
    interior nodes when possible, so the conservation fix does not move a
    clipped boundary of the physical interval.  It is used after *every*
    autoregressive forecast step, never with future target data.
    """

    if value.ndim != 4 or reference_mass.shape != (*value.shape[:2], 1, 1):
        raise ValueError("value must be [batch,channels,H,W] and reference_mass [batch,channels,1,1].")
    if not lower < upper or interior_margin < 0.0:
        raise ValueError("Invalid physical bounds or interior margin.")
    bounded = value.clamp(lower, upper)
    weights = trapezoidal_node_weights(bounded)
    interior = (bounded > lower + interior_margin) & (bounded < upper - interior_margin)
    interior_weight = (weights * interior).sum(dim=(-2, -1), keepdim=True)
    all_weight = weights.sum(dim=(-2, -1), keepdim=True)
    usable = interior_weight > torch.finfo(bounded.dtype).eps
    correction_weight = torch.where(usable, interior_weight, all_weight)
    correction_mask = torch.where(usable, interior, torch.ones_like(interior))
    mass_error = trapezoidal_mass(bounded) - reference_mass
    projected = bounded - correction_mask.to(dtype=bounded.dtype) * mass_error / correction_weight
    # The correction is tiny for V3.1's solver-consistent states.  Clamp once
    # more to retain the declared physical range; diagnostics record any
    # residual mass change due to a truly saturated prediction.
    return projected.clamp(lower, upper)


def bulk_trapezoidal_projection_mask(
    value: torch.Tensor,
    *,
    lower: float = -0.99,
    upper: float = 0.99,
    interior_margin: float = 0.005,
    bulk_threshold: float = 0.35,
) -> torch.Tensor:
    """Select causal, non-interface nodes eligible for a mass correction.

    The diffuse interface is deliberately *not* eligible: it is identified
    solely from the current prediction, not from a future target.  This is a
    diagnostic/replay primitive, rather than a claim that bulk re-centering is
    a physical substitute for an interface-resolved flux update.
    """

    if value.ndim != 4:
        raise ValueError("bulk_trapezoidal_projection_mask requires [batch,channels,H,W].")
    if not lower < upper or interior_margin < 0.0 or not 0.0 < bulk_threshold < upper:
        raise ValueError("Invalid bounds, interior margin, or bulk threshold.")
    bounded = value.clamp(lower, upper)
    return (
        (bounded.abs() >= bulk_threshold)
        & (bounded > lower + interior_margin)
        & (bounded < upper - interior_margin)
    )


def project_trapezoidal_mass_in_bulk(
    value: torch.Tensor,
    reference_mass: torch.Tensor,
    *,
    lower: float = -0.99,
    upper: float = 0.99,
    interior_margin: float = 0.005,
    bulk_threshold: float = 0.35,
    iterations: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project mass only through predicted bulk nodes, never interface nodes.

    A bounded bisection (water-filling) solves the source-grid trapezoidal
    mass constraint exactly when the eligible bulk has sufficient capacity.
    It deliberately refuses an infeasible correction instead of silently
    falling back to an all-pixel projection that would alter the interface.
    The second return value is the causal eligible-node mask.
    """

    if value.ndim != 4 or reference_mass.shape != (*value.shape[:2], 1, 1):
        raise ValueError("value must be [batch,channels,H,W] and reference_mass [batch,channels,1,1].")
    if iterations < 8:
        raise ValueError("Bulk projection requires at least eight bisection iterations.")
    bounded = value.clamp(lower, upper)
    mask = bulk_trapezoidal_projection_mask(
        bounded,
        lower=lower,
        upper=upper,
        interior_margin=interior_margin,
        bulk_threshold=bulk_threshold,
    )
    weights = trapezoidal_node_weights(bounded)
    mask_float = mask.to(dtype=bounded.dtype)
    fixed = bounded * (1.0 - mask_float)
    min_mass = (fixed + lower * mask_float).mul(weights).sum(dim=(-2, -1), keepdim=True)
    max_mass = (fixed + upper * mask_float).mul(weights).sum(dim=(-2, -1), keepdim=True)
    tolerance = 32.0 * torch.finfo(bounded.dtype).eps * weights.sum(dim=(-2, -1), keepdim=True)
    feasible = (reference_mass >= min_mass - tolerance) & (reference_mass <= max_mass + tolerance)
    if not bool(feasible.all()):
        failed = int((~feasible).sum().item())
        raise RuntimeError(
            f"Bulk-only source-grid mass correction is infeasible for {failed} sample/channel fields; "
            "refusing an interface-changing fallback."
        )

    # p(lambda) = clamp(value - lambda) is monotone decreasing in lambda.
    # These limits cover the full declared physical range for every node.
    lo = torch.full_like(reference_mass, lower - upper)
    hi = torch.full_like(reference_mass, upper - lower)
    for _ in range(iterations):
        midpoint = 0.5 * (lo + hi)
        candidate = fixed + mask_float * (bounded - midpoint).clamp(lower, upper)
        mass = (candidate * weights).sum(dim=(-2, -1), keepdim=True)
        lo = torch.where(mass > reference_mass, midpoint, lo)
        hi = torch.where(mass > reference_mass, hi, midpoint)
    projected = fixed + mask_float * (bounded - 0.5 * (lo + hi)).clamp(lower, upper)
    return projected, mask


def neumann_conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Conv2d:
    """Return an odd-kernel convolution with zero-flux-compatible edge padding."""

    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("neumann_conv requires a positive odd kernel size.")
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="replicate",
    )


def _groups(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class NeumannResidualBlock(nn.Module):
    """Local residual block that never connects opposite domain edges."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(_groups(channels), channels)
        self.conv_1 = neumann_conv(channels, channels)
        self.norm_2 = nn.GroupNorm(_groups(channels), channels)
        self.conv_2 = neumann_conv(channels, channels)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.conv_1(self.activation(self.norm_1(value)))
        update = self.conv_2(self.activation(self.norm_2(update)))
        return value + update


class ConditionedNeumannResidualBlock(nn.Module):
    """Neumann residual block with bounded per-sample FiLM conditioning."""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(_groups(channels), channels)
        self.conv_1 = neumann_conv(channels, channels)
        self.norm_2 = nn.GroupNorm(_groups(channels), channels)
        self.conv_2 = neumann_conv(channels, channels)
        self.style_1 = nn.Linear(condition_channels, 2 * channels)
        self.style_2 = nn.Linear(condition_channels, 2 * channels)
        self.activation = nn.SiLU()

    @staticmethod
    def _modulate(value: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        scale, shift = affine.chunk(2, dim=1)
        scale = 0.20 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = 0.20 * torch.tanh(shift).unsqueeze(-1).unsqueeze(-1)
        return value * (1.0 + scale) + shift

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != value.shape[0]:
            raise ValueError("Condition must be [batch, features] for a conditioned Neumann residual block.")
        update = self._modulate(self.norm_1(value), self.style_1(condition))
        update = self.conv_1(self.activation(update))
        update = self._modulate(self.norm_2(update), self.style_2(condition))
        update = self.conv_2(self.activation(update))
        return value + update


def zero_flux_divergence(potential: torch.Tensor, mobility: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``div(M grad(q))`` with zero normal flux at all four boundaries.

    ``potential`` and optional positive ``mobility`` are [B,C,H,W].  Face
    fluxes exist only between adjacent cells; no `roll`, circular padding, or
    cross-boundary connection is used.  Unlike an after-the-fact global
    re-centring, this preserves the local face-flux interpretation all the
    way to the boundary; its spatial sum is zero in exact arithmetic.
    """

    if potential.ndim != 4:
        raise ValueError("zero_flux_divergence expects [batch, channels, height, width].")
    if potential.shape[-2] < 2 or potential.shape[-1] < 2:
        raise ValueError("zero_flux_divergence requires spatial dimensions of at least two.")
    if mobility is not None and mobility.shape != potential.shape:
        raise ValueError("mobility must have exactly the potential shape.")

    horizontal = potential[..., 1:] - potential[..., :-1]
    vertical = potential[..., 1:, :] - potential[..., :-1, :]
    if mobility is not None:
        horizontal = horizontal * 0.5 * (mobility[..., 1:] + mobility[..., :-1])
        vertical = vertical * 0.5 * (mobility[..., 1:, :] + mobility[..., :-1, :])

    divergence = torch.zeros_like(potential)
    divergence[..., 0] += horizontal[..., 0]
    divergence[..., 1:-1] += horizontal[..., 1:] - horizontal[..., :-1]
    divergence[..., -1] -= horizontal[..., -1]
    divergence[..., 0, :] += vertical[..., 0, :]
    divergence[..., 1:-1, :] += vertical[..., 1:, :] - vertical[..., :-1, :]
    divergence[..., -1, :] -= vertical[..., -1, :]
    return divergence


def trapezoidal_zero_flux_divergence(potential: torch.Tensor, mobility: torch.Tensor | None = None) -> torch.Tensor:
    """Return a zero-normal-flux update conserving trapezoidal node mass.

    ``zero_flux_divergence`` is a cell-centred face-flux stencil, hence its
    *unweighted* sum telescopes to zero.  VCH-v3.1 is stored on the solver's
    node grid and audits ``sum(W * phi)`` with trapezoidal weights ``W``.
    Dividing the face divergence by ``W`` yields the corresponding local
    node-grid update: ``sum(W * update)`` telescopes exactly, while no
    opposite edges are connected and no global after-the-fact recentering is
    used.
    """

    divergence = zero_flux_divergence(potential, mobility)
    return divergence / trapezoidal_node_weights(divergence)


def neumann_interface_strength(value: torch.Tensor) -> torch.Tensor:
    """Forward-difference interface magnitude with no periodic edge seam."""

    if value.ndim != 4:
        raise ValueError("neumann_interface_strength expects [batch, channels, height, width].")
    dx = torch.zeros_like(value)
    dy = torch.zeros_like(value)
    dx[..., :-1] = value[..., 1:] - value[..., :-1]
    dy[..., :-1, :] = value[..., 1:, :] - value[..., :-1, :]
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return magnitude / magnitude.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
