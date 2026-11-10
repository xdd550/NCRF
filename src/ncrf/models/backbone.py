"""V15 NCRF: Neumann-aware control-relaxation flux forecaster.

This is a structure-informed long-horizon forecaster, not a replacement PDE
solver.  It accepts only ten past phi/u frames and prescribed future u.  The
three independently switchable components are deliberately falsifiable:

* N0: Neumann-local autoregressive backbone;
* N1: causal latent control-relaxation state;
* N2: zero-normal-flux conservative morphology update;
* N3: both N1 and N2.

No future phi, chemical potential mu, or unobserved relaxation state w enters
the model.  The relaxation state below is a learned latent initialized from
past controls; it must never be presented as an observed physical w field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from .operators import (
    ConditionedNeumannResidualBlock,
    NeumannResidualBlock,
    project_trapezoidal_mass_in_bulk,
    neumann_conv,
    neumann_interface_strength,
    project_trapezoidal_mass,
    trapezoidal_mass,
    trapezoidal_node_weights,
    zero_flux_divergence,
)


@dataclass(frozen=True)
class VCHV15Config:
    """Capacity, numerical bounds, and explicit N0--N3 ablation switches."""

    input_steps: int = 10
    image_size: int = 128
    hidden_channels: int = 32
    fine_blocks: int = 2
    coarse_blocks: int = 2
    downsample_factor: int = 2
    # None preserves the historical rollout-relative time condition.  A
    # positive fixed horizon makes the same physical forecast step receive
    # the same condition in H10, H20, H50, and H90 rollouts.
    time_coordinate_horizon: int | None = None
    max_local_step_delta: float = 0.16
    max_global_mean_step_delta: float = 0.004
    use_control_relaxation: bool = True
    use_conservative_flux: bool = True
    use_adaptive_flux_mobility: bool = False
    use_causal_interface_transport: bool = False
    use_source_grid_mass_projection: bool = False
    use_source_grid_mass_bulk_projection: bool = False
    source_grid_projection_lower_bound: float = -0.99
    source_grid_projection_upper_bound: float = 0.99
    source_grid_projection_interior_margin: float = 0.005
    source_grid_bulk_projection_threshold: float = 0.35
    interface_transport_mode: str = "interface_temporal"
    control_relaxation_gain: float = 0.06
    flux_residual_gain: float = 0.10
    adaptive_flux_gate_amplitude: float = 0.25
    interface_transport_flux_gain: float = 0.06
    interface_transport_late_start_fraction: float = 1.0 / 3.0
    relaxation_dt: float = 0.01
    relaxation_gamma: float = 10.0
    relaxation_scheme: str = "crank_nicolson_known_control"

    def __post_init__(self) -> None:
        if self.input_steps != 10:
            raise ValueError("The registered VCH protocol requires exactly ten context frames.")
        if self.image_size < 16 or self.image_size % self.downsample_factor != 0:
            raise ValueError("image_size must be at least 16 and divisible by downsample_factor.")
        if self.hidden_channels < 8 or self.fine_blocks < 1 or self.coarse_blocks < 1:
            raise ValueError("V15 widths and block counts must be positive.")
        if self.downsample_factor not in {2, 4}:
            raise ValueError("V15 currently supports a 2x or 4x coarse path.")
        if self.time_coordinate_horizon is not None and (
            type(self.time_coordinate_horizon) is not int or self.time_coordinate_horizon < 1
        ):
            raise ValueError("time_coordinate_horizon must be a positive integer or None.")
        if self.max_local_step_delta <= 0.0 or self.max_global_mean_step_delta <= 0.0:
            raise ValueError("Autoregressive residual bounds must be positive.")
        if not 0.0 <= self.control_relaxation_gain <= 0.20:
            raise ValueError("control_relaxation_gain must lie in [0, 0.20].")
        if not 0.0 <= self.flux_residual_gain <= 0.25:
            raise ValueError("flux_residual_gain must lie in [0, 0.25].")
        if not 0.0 <= self.adaptive_flux_gate_amplitude <= 0.50:
            raise ValueError("adaptive_flux_gate_amplitude must lie in [0, 0.50].")
        if self.use_adaptive_flux_mobility and not self.use_conservative_flux:
            raise ValueError("Adaptive flux mobility requires the conservative flux update.")
        if self.use_causal_interface_transport and not self.use_conservative_flux:
            raise ValueError("Causal interface transport requires the conservative flux update.")
        if (self.use_source_grid_mass_projection or self.use_source_grid_mass_bulk_projection) and not self.use_conservative_flux:
            raise ValueError("Source-grid mass projection is registered only for the N2 conservative-flux backbone.")
        if self.use_source_grid_mass_projection and self.use_source_grid_mass_bulk_projection:
            raise ValueError("Global and bulk-only source-grid projections are mutually exclusive.")
        if not self.source_grid_projection_lower_bound < self.source_grid_projection_upper_bound:
            raise ValueError("source-grid projection bounds must be ordered.")
        if self.source_grid_projection_interior_margin < 0.0:
            raise ValueError("source_grid_projection_interior_margin must be non-negative.")
        if not 0.20 < self.source_grid_bulk_projection_threshold < self.source_grid_projection_upper_bound:
            raise ValueError("source_grid_bulk_projection_threshold must exclude the registered ±0.20 interface band.")
        if not 0.0 <= self.interface_transport_flux_gain <= 0.15:
            raise ValueError("interface_transport_flux_gain must lie in [0, 0.15].")
        if not 0.0 <= self.interface_transport_late_start_fraction < 1.0:
            raise ValueError("interface_transport_late_start_fraction must lie in [0, 1).")
        if self.interface_transport_mode not in {"interface_temporal", "generic_late_flux"}:
            raise ValueError("interface_transport_mode must be 'interface_temporal' or 'generic_late_flux'.")
        if not 0.0 < self.relaxation_dt <= 0.1 or not 0.0 < self.relaxation_gamma <= 100.0:
            raise ValueError("relaxation_dt and relaxation_gamma must be positive within registered bounds.")
        if self.relaxation_scheme != "crank_nicolson_known_control":
            raise ValueError("V15 only supports the registered crank_nicolson_known_control relaxation scheme.")


class VCHNeumannControlRelaxationFluxForecaster(nn.Module):
    """Neumann-local multi-scale recurrence with optional causal/flux updates."""

    def __init__(self, config: VCHV15Config) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_channels
        context_channels = 2 * config.input_steps

        self.context_fine = nn.Sequential(
            neumann_conv(context_channels, width), nn.SiLU(), NeumannResidualBlock(width), NeumannResidualBlock(width)
        )
        self.context_coarse = nn.Sequential(neumann_conv(context_channels, width), nn.SiLU(), NeumannResidualBlock(width))
        self.value_encoder = nn.Sequential(neumann_conv(1, width), nn.SiLU(), NeumannResidualBlock(width))
        self.control_encoder = nn.Sequential(neumann_conv(1, width), nn.SiLU(), NeumannResidualBlock(width))
        self.control_conditioner = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width), nn.Tanh())

        self.coarse_input = nn.Sequential(neumann_conv(3 * width, width), nn.SiLU())
        self.coarse_blocks = nn.ModuleList(ConditionedNeumannResidualBlock(width, width) for _ in range(config.coarse_blocks))
        self.coarse_delta = nn.Sequential(neumann_conv(width, width), nn.SiLU(), neumann_conv(width, 1))
        self.fine_input = nn.Sequential(neumann_conv(4 * width, width), nn.SiLU())
        self.fine_blocks = nn.ModuleList(ConditionedNeumannResidualBlock(width, width) for _ in range(config.fine_blocks))
        self.fine_delta = nn.Sequential(neumann_conv(width, width), nn.SiLU(), neumann_conv(width, 1))

        # The latent memory is initialized solely from known context controls,
        # then updated using the declared gamma*dw/dt + w = u relaxation form.
        self.relaxation_initializer = nn.Sequential(
            neumann_conv(config.input_steps, width), nn.SiLU(), NeumannResidualBlock(width), neumann_conv(width, 1), nn.Tanh()
        )
        self.relaxation_encoder = nn.Sequential(neumann_conv(1, width), nn.SiLU(), NeumannResidualBlock(width))
        self.relaxation_delta = nn.Sequential(neumann_conv(2 * width, width), nn.SiLU(), neumann_conv(width, 1))

        # q and M are latent learned quantities, not the unavailable chemical
        # potential/mobility.  The helper turns them into a true zero-flux
        # divergence update without periodic wraparound.
        self.flux_potential = nn.Sequential(
            neumann_conv(2 * width, width), nn.SiLU(), NeumannResidualBlock(width), neumann_conv(width, 1)
        )
        self.flux_mobility = nn.Sequential(neumann_conv(width, width), nn.SiLU(), neumann_conv(width, 1), nn.Sigmoid())

        self.mean_drift_head = nn.Sequential(nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.fine_feedback = nn.Sequential(neumann_conv(1, width), nn.SiLU(), NeumannResidualBlock(width))
        self.coarse_feedback = nn.Sequential(neumann_conv(1, width), nn.SiLU(), NeumannResidualBlock(width))
        self.fine_gate = nn.Conv2d(3 * width, width, 1)
        self.fine_candidate = nn.Conv2d(3 * width, width, 1)
        self.coarse_gate = nn.Conv2d(3 * width, width, 1)
        self.coarse_candidate = nn.Conv2d(3 * width, width, 1)

        if config.use_adaptive_flux_mobility:
            # Register this only after every original N2 module has been
            # initialized.  With the same seed, the shared N2 parameters are
            # therefore bitwise identical, while the zero-initialized final
            # gate head makes the initial forward pass exactly N2 (gate=1).
            self.flux_mobility_gate: nn.Sequential | None = nn.Sequential(
                neumann_conv(2 * width, width), nn.SiLU(), neumann_conv(width, 1)
            )
            gate_head = self.flux_mobility_gate[-1]
            assert isinstance(gate_head, nn.Conv2d)
            nn.init.zeros_(gate_head.weight)
            nn.init.zeros_(gate_head.bias)
        else:
            self.flux_mobility_gate = None

        if config.use_causal_interface_transport:
            # The new head sees only causal state differences, predicted
            # interface strength, and known control.  It is registered after
            # all frozen-N2 modules, and its final potential head is exactly
            # zero, so the initial forward pass is bitwise N2.  The learned
            # scalar is a latent transport potential, not an observed velocity
            # or chemical potential.
            transport_channels = 2 * width + 2
            self.interface_transport_potential: nn.Sequential | None = nn.Sequential(
                neumann_conv(transport_channels, width), nn.SiLU(), NeumannResidualBlock(width), neumann_conv(width, 1)
            )
            self.interface_transport_mobility: nn.Sequential | None = nn.Sequential(
                neumann_conv(transport_channels, width), nn.SiLU(), neumann_conv(width, 1), nn.Sigmoid()
            )
            transport_head = self.interface_transport_potential[-1]
            assert isinstance(transport_head, nn.Conv2d)
            nn.init.zeros_(transport_head.weight)
            nn.init.zeros_(transport_head.bias)
        else:
            self.interface_transport_potential = None
            self.interface_transport_mobility = None

    @property
    def relaxation_memory_coefficient(self) -> float:
        """Crank--Nicolson coefficient multiplying the previous latent state."""

        half_step = 0.5 * self.config.relaxation_dt
        return float((self.config.relaxation_gamma - half_step) / (self.config.relaxation_gamma + half_step))

    @property
    def relaxation_control_coefficient(self) -> float:
        """Crank--Nicolson drive coefficient applied to the mean of adjacent u frames."""

        return float(self.config.relaxation_dt / (self.config.relaxation_gamma + 0.5 * self.config.relaxation_dt))

    def _validate(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> None:
        expected_context = (self.config.input_steps, 1, self.config.image_size, self.config.image_size)
        expected_field = (1, self.config.image_size, self.config.image_size)
        if context_phi.ndim != 5 or tuple(context_phi.shape[1:]) != expected_context:
            raise ValueError(f"Expected context_phi [batch,{expected_context}], got {tuple(context_phi.shape)}.")
        if context_u.shape != context_phi.shape:
            raise ValueError("context_u must have exactly the context_phi shape.")
        if future_u.ndim != 5 or future_u.shape[0] != context_phi.shape[0] or tuple(future_u.shape[2:]) != expected_field:
            raise ValueError(f"Expected future_u [batch,horizon,{expected_field}], got {tuple(future_u.shape)}.")
        if future_u.shape[1] < 1:
            raise ValueError("future_u must contain at least one prescribed control frame.")
        if not all(bool(torch.isfinite(item).all()) for item in (context_phi, context_u, future_u)):
            raise ValueError("VCH V15 inputs must be finite.")

    def _pool(self, value: torch.Tensor) -> torch.Tensor:
        return functional.avg_pool2d(value, self.config.downsample_factor, self.config.downsample_factor)

    def _initial_state(
        self, context_phi: torch.Tensor, context_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked = torch.cat((context_phi.flatten(start_dim=1, end_dim=2), context_u.flatten(start_dim=1, end_dim=2)), dim=1)
        fine = self.context_fine(stacked)
        coarse = self.context_coarse(self._pool(stacked))
        relaxation = self.relaxation_initializer(context_u.flatten(start_dim=1, end_dim=2))
        return fine, coarse, relaxation

    def _control_features(self, control: torch.Tensor, progress: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.control_encoder(control)
        stats = torch.cat(
            (
                control.float().mean(dim=(-2, -1)),
                control.float().std(dim=(-2, -1), unbiased=False),
                torch.full((control.shape[0], 1), progress, dtype=torch.float32, device=control.device),
            ),
            dim=1,
        )
        return local, self._pool(local), self.control_conditioner(stats.to(dtype=local.dtype))

    @staticmethod
    def _zero_mean(value: torch.Tensor) -> torch.Tensor:
        return value.float() - value.float().mean(dim=(-2, -1), keepdim=True)

    @staticmethod
    def _update_hidden(
        hidden: torch.Tensor, dynamics: torch.Tensor, feedback: torch.Tensor, gate_layer: nn.Module, candidate_layer: nn.Module
    ) -> torch.Tensor:
        update_input = torch.cat((hidden, dynamics, feedback), dim=1)
        gate = torch.sigmoid(gate_layer(update_input))
        candidate = torch.tanh(candidate_layer(update_input))
        return gate * hidden + (1.0 - gate) * candidate

    def _structured_updates(
        self,
        fine_dynamics: torch.Tensor,
        relaxation: torch.Tensor,
        control_features: torch.Tensor,
        value: torch.Tensor,
        previous_value: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return relaxation plus conservative base and auxiliary fluxes.

        The auxiliary branch is a raw, zero-normal-flux divergence.  It is
        applied after the nonlinear local residual in ``_step`` so its field
        contribution can be measured separately from the base local update.
        """

        relaxation_features = self.relaxation_encoder(relaxation)
        zeros = torch.zeros_like(relaxation)
        if self.config.use_control_relaxation:
            relaxation_update = self.config.control_relaxation_gain * torch.tanh(
                self.relaxation_delta(torch.cat((fine_dynamics, relaxation_features), dim=1))
            )
        else:
            relaxation_update = zeros
        if self.config.use_conservative_flux:
            potential = torch.tanh(self.flux_potential(torch.cat((fine_dynamics, relaxation_features), dim=1)))
            mobility = self.flux_mobility(fine_dynamics).to(dtype=potential.dtype)
            if self.flux_mobility_gate is not None:
                gate_input = torch.cat((fine_dynamics, control_features), dim=1)
                flux_mobility_gate = 1.0 + self.config.adaptive_flux_gate_amplitude * torch.tanh(
                    self.flux_mobility_gate(gate_input)
                )
                mobility = mobility * flux_mobility_gate.to(dtype=mobility.dtype)
            else:
                flux_mobility_gate = torch.ones_like(mobility)
            flux_update = self.config.flux_residual_gain * zero_flux_divergence(potential, mobility)
        else:
            flux_update = zeros
            flux_mobility_gate = torch.ones_like(zeros)
        if self.interface_transport_potential is not None and self.interface_transport_mobility is not None:
            interface_strength = neumann_interface_strength(value)
            interface_activity = interface_strength / (1.0 + interface_strength)
            temporal_motion = value - previous_value
            if self.config.interface_transport_mode == "interface_temporal":
                transport_input = torch.cat((fine_dynamics, control_features, temporal_motion, interface_activity), dim=1)
                transport_mobility_gate = interface_activity
            else:
                # Capacity-matched control: same trainable branch and late
                # schedule, but neither explicit temporal difference nor
                # interface localization reaches the branch. Use real
                # current-state/control channels so no input weights are
                # artificially inactive. It cannot support a
                # causal-interface mechanism claim by itself.
                transport_input = torch.cat((fine_dynamics, control_features, value, control_features.mean(dim=1, keepdim=True)), dim=1)
                transport_mobility_gate = torch.ones_like(interface_activity)
            transport_potential = torch.tanh(self.interface_transport_potential(transport_input))
            transport_mobility = transport_mobility_gate * self.interface_transport_mobility(transport_input).to(dtype=value.dtype)
            late_ramp = max(
                0.0,
                min(1.0, (progress - self.config.interface_transport_late_start_fraction) / (1.0 - self.config.interface_transport_late_start_fraction)),
            )
            interface_transport_update = (
                self.config.interface_transport_flux_gain * late_ramp * zero_flux_divergence(transport_potential, transport_mobility)
            )
        else:
            interface_transport_update = zeros
            interface_activity = torch.zeros_like(zeros)
        return (
            self._zero_mean(relaxation_update),
            flux_update,
            relaxation_features,
            flux_mobility_gate,
            interface_transport_update,
            interface_activity,
        )

    def _step(
        self,
        value: torch.Tensor,
        previous_value: torch.Tensor,
        fine_hidden: torch.Tensor,
        coarse_hidden: torch.Tensor,
        relaxation: torch.Tensor,
        control: torch.Tensor,
        previous_control: torch.Tensor,
        progress: float,
        reference_mass: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        control_fine, control_coarse, condition = self._control_features(control, progress)
        value_feature = self.value_encoder(value)
        coarse_value_feature = self._pool(value_feature)
        coarse_dynamics = self.coarse_input(torch.cat((coarse_value_feature, coarse_hidden, control_coarse), dim=1))
        for block in self.coarse_blocks:
            coarse_dynamics = block(coarse_dynamics, condition)
        coarse_delta = self._zero_mean(self.coarse_delta(coarse_dynamics))
        coarse_features_up = functional.interpolate(coarse_dynamics, size=value.shape[-2:], mode="bilinear", align_corners=False)
        coarse_delta_up = functional.interpolate(coarse_delta, size=value.shape[-2:], mode="bilinear", align_corners=False)

        fine_dynamics = self.fine_input(torch.cat((value_feature, fine_hidden, control_fine, coarse_features_up), dim=1))
        for block in self.fine_blocks:
            fine_dynamics = block(fine_dynamics, condition)

        # A CN-consistent control update.  The initial field remains learned
        # from context controls, because the true w is deliberately withheld.
        next_relaxation = (
            self.relaxation_memory_coefficient * relaxation
            + self.relaxation_control_coefficient * 0.5 * (previous_control + control)
        )
        relaxation_update, base_flux_update, _, flux_mobility_gate, interface_transport_update, interface_activity = self._structured_updates(
            fine_dynamics, next_relaxation, control_fine, value, previous_value, progress
        )
        # Keep the registered N2 flux inside the original bounded local
        # residual.  The new auxiliary branch is intentionally kept outside
        # this tanh so its output-level contribution can be audited rather
        # than being silently mixed into a nonlinear latent residual.
        total_flux_update = base_flux_update + interface_transport_update
        local_delta = self._zero_mean(self.fine_delta(fine_dynamics) + coarse_delta_up + relaxation_update + base_flux_update)

        control_mean = control.float().mean(dim=(-2, -1))
        control_std = control.float().std(dim=(-2, -1), unbiased=False)
        mean_features = torch.cat(
            (
                fine_dynamics.float().mean(dim=(-2, -1)),
                coarse_dynamics.float().mean(dim=(-2, -1)),
                control_mean,
                control_std,
                torch.full((value.shape[0], 1), progress, dtype=torch.float32, device=value.device),
            ),
            dim=1,
        )
        mean_delta = self.config.max_global_mean_step_delta * torch.tanh(self.mean_drift_head(mean_features)).view(-1, 1, 1, 1)
        base_pre_bound = value.float() + self.config.max_local_step_delta * torch.tanh(local_delta) + mean_delta
        candidate_pre_bound = base_pre_bound + interface_transport_update.float()
        base_prediction = base_pre_bound.clamp(-1.0, 1.0)
        candidate_prediction = candidate_pre_bound.clamp(-1.0, 1.0)
        # Every projection is deterministic and parameter-free.  It is
        # applied before recurrent feedback is encoded, so this closed-loop
        # path can be screened honestly with a frozen A2 checkpoint.  The
        # target phi never reaches this path.
        if reference_mass is None:
            prediction = candidate_prediction
            source_grid_projection_update = torch.zeros_like(prediction)
            bulk_projection_fraction = torch.zeros_like(prediction[..., :1, :1])
            bulk_interface_correction_max = torch.zeros_like(prediction[..., :1, :1])
        elif self.config.use_source_grid_mass_bulk_projection:
            prediction, bulk_mask = project_trapezoidal_mass_in_bulk(
                candidate_prediction,
                reference_mass,
                lower=self.config.source_grid_projection_lower_bound,
                upper=self.config.source_grid_projection_upper_bound,
                interior_margin=self.config.source_grid_projection_interior_margin,
                bulk_threshold=self.config.source_grid_bulk_projection_threshold,
            )
            source_grid_projection_update = prediction - candidate_prediction
            bulk_projection_fraction = bulk_mask.float().mean(dim=(-2, -1), keepdim=True)
            interface = candidate_prediction.abs() < self.config.source_grid_bulk_projection_threshold
            bulk_interface_correction_max = (source_grid_projection_update.abs() * interface).amax(
                dim=(-2, -1), keepdim=True
            )
        else:
            prediction = project_trapezoidal_mass(
                candidate_prediction,
                reference_mass,
                lower=self.config.source_grid_projection_lower_bound,
                upper=self.config.source_grid_projection_upper_bound,
                interior_margin=self.config.source_grid_projection_interior_margin,
            )
            source_grid_projection_update = prediction - candidate_prediction
            bulk_projection_fraction = torch.zeros_like(prediction[..., :1, :1])
            bulk_interface_correction_max = torch.zeros_like(prediction[..., :1, :1])
        transport_output_update = candidate_prediction - base_prediction
        transport_clamp_affected_fraction = (
            (interface_transport_update != 0)
            & ((base_pre_bound.abs() > 1.0) | (candidate_pre_bound.abs() > 1.0))
        ).float().mean(dim=(-2, -1), keepdim=True)

        fine_feedback = self.fine_feedback(prediction.to(dtype=value.dtype))
        coarse_feedback = self.coarse_feedback(self._pool(prediction).to(dtype=value.dtype))
        next_fine = self._update_hidden(fine_hidden, fine_dynamics, fine_feedback, self.fine_gate, self.fine_candidate)
        next_coarse = self._update_hidden(coarse_hidden, coarse_dynamics, coarse_feedback, self.coarse_gate, self.coarse_candidate)
        return (
            prediction,
            next_fine,
            next_coarse,
            next_relaxation,
            mean_delta,
            total_flux_update,
            relaxation_update,
            flux_mobility_gate,
            interface_transport_update,
            interface_activity,
            transport_output_update,
            transport_clamp_affected_fraction,
            source_grid_projection_update,
            bulk_projection_fraction,
            bulk_interface_correction_max,
        )

    def forecast(
        self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor, return_diagnostics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forecast using only declared causal inputs and optional diagnostics."""

        self._validate(context_phi, context_u, future_u)
        value = context_phi[:, -1]
        previous_value = context_phi[:, -2]
        previous_control = context_u[:, -1]
        fine_hidden, coarse_hidden, relaxation = self._initial_state(context_phi, context_u)
        reference_mass = (
            trapezoidal_mass(context_phi[:, -1].float())
            if self.config.use_source_grid_mass_projection or self.config.use_source_grid_mass_bulk_projection
            else None
        )
        predictions, mean_deltas, flux_magnitudes, flux_sums, relaxation_magnitudes, flux_gate_means = [], [], [], [], [], []
        transport_magnitudes, transport_sums, interface_activity_means = [], [], []
        transport_output_magnitudes, transport_output_sums, transport_clamp_fractions = [], [], []
        projection_magnitudes, source_mass_errors, bulk_projection_fractions, bulk_interface_correction_maxima = [], [], [], []
        horizon = future_u.shape[1]
        progress_horizon = self.config.time_coordinate_horizon or horizon
        if horizon > progress_horizon:
            raise ValueError(
                f"Requested horizon {horizon} exceeds fixed time_coordinate_horizon {progress_horizon}."
            )
        for step in range(horizon):
            current_value = value
            (
                value,
                fine_hidden,
                coarse_hidden,
                relaxation,
                mean_delta,
                flux_update,
                relaxation_update,
                flux_mobility_gate,
                interface_transport_update,
                interface_activity,
                transport_output_update,
                transport_clamp_affected_fraction,
                source_grid_projection_update,
                bulk_projection_fraction,
                bulk_interface_correction_max,
            ) = self._step(
                value,
                previous_value,
                fine_hidden,
                coarse_hidden,
                relaxation,
                future_u[:, step],
                previous_control,
                progress=(step + 1) / progress_horizon,
                reference_mass=reference_mass,
            )
            predictions.append(value)
            mean_deltas.append(mean_delta)
            flux_magnitudes.append(flux_update.abs().mean(dim=(-2, -1)))
            flux_sums.append(flux_update.sum(dim=(-2, -1)))
            relaxation_magnitudes.append(relaxation_update.abs().mean(dim=(-2, -1)))
            flux_gate_means.append(flux_mobility_gate.mean(dim=(-2, -1)))
            transport_magnitudes.append(interface_transport_update.abs().mean(dim=(-2, -1)))
            transport_sums.append(interface_transport_update.sum(dim=(-2, -1)))
            interface_activity_means.append(interface_activity.mean(dim=(-2, -1)))
            transport_output_magnitudes.append(transport_output_update.abs().mean(dim=(-2, -1)))
            transport_output_sums.append(transport_output_update.sum(dim=(-2, -1)))
            transport_clamp_fractions.append(transport_clamp_affected_fraction)
            projection_magnitudes.append(source_grid_projection_update.abs().mean(dim=(-2, -1)))
            bulk_projection_fractions.append(bulk_projection_fraction.squeeze(-1).squeeze(-1))
            bulk_interface_correction_maxima.append(bulk_interface_correction_max.squeeze(-1).squeeze(-1))
            if reference_mass is None:
                source_mass_errors.append(torch.zeros_like(value[..., :1, :1]))
            else:
                source_area = trapezoidal_node_weights(value).sum(dim=(-2, -1), keepdim=True)
                source_mass_errors.append((trapezoidal_mass(value.float()) - reference_mass) / source_area)
            previous_control = future_u[:, step]
            previous_value = current_value
        output = torch.stack(predictions, dim=1)
        if not return_diagnostics:
            return output
        return output, {
            "predicted_mean_drift_from_last_context": output.mean(dim=(-2, -1)
            ) - context_phi[:, -1].mean(dim=(-2, -1)).unsqueeze(1),
            "mean_delta_per_step": torch.stack(mean_deltas, dim=1).squeeze(-1).squeeze(-1),
            "mean_flux_abs_per_step": torch.stack(flux_magnitudes, dim=1).squeeze(-1).squeeze(-1),
            "flux_spatial_sum_per_step": torch.stack(flux_sums, dim=1).squeeze(-1).squeeze(-1),
            "mean_flux_mobility_gate_per_step": torch.stack(flux_gate_means, dim=1).squeeze(-1).squeeze(-1),
            "mean_interface_transport_flux_abs_per_step": torch.stack(transport_magnitudes, dim=1).squeeze(-1).squeeze(-1),
            "interface_transport_flux_spatial_sum_per_step": torch.stack(transport_sums, dim=1).squeeze(-1).squeeze(-1),
            "mean_interface_transport_activity_per_step": torch.stack(interface_activity_means, dim=1).squeeze(-1).squeeze(-1),
            "mean_interface_transport_output_abs_per_step": torch.stack(transport_output_magnitudes, dim=1).squeeze(-1).squeeze(-1),
            "interface_transport_output_spatial_sum_per_step": torch.stack(transport_output_sums, dim=1).squeeze(-1).squeeze(-1),
            "interface_transport_clamp_affected_fraction_per_step": torch.stack(transport_clamp_fractions, dim=1).squeeze(-1).squeeze(-1),
            "mean_relaxation_update_abs_per_step": torch.stack(relaxation_magnitudes, dim=1).squeeze(-1).squeeze(-1),
            "mean_source_grid_projection_abs_per_step": torch.stack(projection_magnitudes, dim=1).squeeze(-1).squeeze(-1),
            "bulk_source_grid_projection_fraction_per_step": torch.stack(bulk_projection_fractions, dim=1).squeeze(-1),
            "bulk_interface_correction_abs_max_per_step": torch.stack(bulk_interface_correction_maxima, dim=1).squeeze(-1),
            "source_grid_mass_error_per_step": torch.stack(source_mass_errors, dim=1).squeeze(-1).squeeze(-1),
        }

    def forward(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> torch.Tensor:
        return self.forecast(context_phi, context_u, future_u)  # type: ignore[return-value]
