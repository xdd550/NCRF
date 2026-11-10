"""V31 controlled NCRF variants. Historical V15 implementation is unchanged.

NoMemory removes only the dedicated control-relaxation memory, NOT general
recurrent memory. GenericUpdate changes the operator, not just conservation;
its gain is fixed a priori, so amplitude/frequency effects are not isolated.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import (
    VCHV15Config, VCHNeumannControlRelaxationFluxForecaster,
)
from .operators import neumann_conv, zero_flux_divergence


class V31NCRF(VCHNeumannControlRelaxationFluxForecaster):
    def __init__(self, config: VCHV15Config, *, variant="full", checkpoint_steps=True):
        if variant not in {"full", "no_memory", "generic_update"}:
            raise ValueError("Unknown V31 NCRF variant.")
        if (config.use_control_relaxation or not config.use_conservative_flux or
            config.use_adaptive_flux_mobility or config.use_causal_interface_transport or
            config.use_source_grid_mass_projection or config.use_source_grid_mass_bulk_projection):
            raise ValueError("V31 core supports only the locked active NCRF backbone.")
        # Initialize the FULL reference first. Deletion cannot shift shared RNG
        # draws. The sliced q weights retain their reference input columns.
        super().__init__(config)
        self.variant = variant
        self.checkpoint_steps = checkpoint_steps
        self.last_step_audit = {}
        if variant == "no_memory":
            self.relaxation_initializer = None
            self.relaxation_encoder = None
            self.relaxation_delta = None
            old = self.flux_potential[0]
            new = copy.deepcopy(old)
            new.in_channels = config.hidden_channels
            new.weight = nn.Parameter(old.weight[:, :config.hidden_channels].detach().clone())
            self.flux_potential[0] = new

    def _initial_state(self, phi, past_u):
        if self.variant != "no_memory":
            return super()._initial_state(phi, past_u)
        stacked = torch.cat((phi.flatten(1, 2), past_u.flatten(1, 2)), dim=1)
        # Placeholder carries no learned memory and is never updated or encoded.
        return self.context_fine(stacked), self.context_coarse(self._pool(stacked)), torch.zeros_like(phi[:, -1])

    def _step(self, *args, **kwargs):
        if self.checkpoint_steps and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._step_impl, *args, **kwargs, use_reentrant=False, preserve_rng_state=True)
        return self._step_impl(*args, **kwargs)

    def _step_impl(self, value, previous_value, fine_hidden, coarse_hidden, relaxation,
                   control, previous_control, progress, reference_mass):
        if reference_mass is not None:
            raise ValueError("No mass projection is part of the V31 core protocol.")
        control_fine, control_coarse, condition = self._control_features(control, progress)
        value_feature = self.value_encoder(value)
        coarse = self.coarse_input(torch.cat((self._pool(value_feature), coarse_hidden, control_coarse), dim=1))
        for block in self.coarse_blocks:
            coarse = block(coarse, condition)
        coarse_delta = self._zero_mean(self.coarse_delta(coarse))
        coarse_up = F.interpolate(coarse, size=value.shape[-2:], mode="bilinear", align_corners=False)
        delta_up = F.interpolate(coarse_delta, size=value.shape[-2:], mode="bilinear", align_corners=False)
        fine = self.fine_input(torch.cat((value_feature, fine_hidden, control_fine, coarse_up), dim=1))
        for block in self.fine_blocks:
            fine = block(fine, condition)
        if self.variant == "no_memory":
            next_r = relaxation  # no CN computation, no initializer/encoder path
            q_input = fine
        else:
            next_r = self.relaxation_memory_coefficient * relaxation + self.relaxation_control_coefficient * 0.5 * (previous_control + control)
            q_input = torch.cat((fine, self.relaxation_encoder(next_r)), dim=1)
        q = torch.tanh(self.flux_potential(q_input))
        mobility = self.flux_mobility(fine).to(q.dtype)
        if self.variant == "generic_update":
            branch = self.config.flux_residual_gain * self._zero_mean(mobility * q)
        else:
            branch = self.config.flux_residual_gain * zero_flux_divergence(q, mobility)
        local = self._zero_mean(self.fine_delta(fine) + delta_up + torch.zeros_like(next_r) + branch)
        stats = torch.cat((fine.float().mean((-2, -1)), coarse.float().mean((-2, -1)),
                           control.float().mean((-2, -1)), control.float().std((-2, -1), unbiased=False),
                           torch.full((value.shape[0], 1), progress, dtype=torch.float32, device=value.device)), dim=1)
        mean_delta = self.config.max_global_mean_step_delta * torch.tanh(self.mean_drift_head(stats)).view(-1, 1, 1, 1)
        pre_bound = value.float() + self.config.max_local_step_delta * torch.tanh(local) + mean_delta
        prediction = pre_bound.clamp(-1, 1)
        with torch.no_grad():
            # Explicitly last-step diagnostics, not trajectory averages.
            self.last_step_audit = {"branch_rms_last_step": float(branch.float().square().mean().sqrt()),
                                    "local_tanh_saturation_last_step": float((local.abs() > 3).float().mean()),
                                    "clip_fraction_last_step": float((pre_bound.abs() > 1).float().mean())}
        next_fine = self._update_hidden(fine_hidden, fine, self.fine_feedback(prediction.to(value.dtype)), self.fine_gate, self.fine_candidate)
        next_coarse = self._update_hidden(coarse_hidden, coarse, self.coarse_feedback(self._pool(prediction).to(value.dtype)), self.coarse_gate, self.coarse_candidate)
        zero = torch.zeros_like(prediction)
        scalar_zero = zero[..., :1, :1]
        return (prediction, next_fine, next_coarse, next_r, mean_delta, branch, zero,
                torch.ones_like(branch), zero, zero, zero, scalar_zero, zero, scalar_zero, scalar_zero)


def core_config(*, size=128):
    return VCHV15Config(image_size=size, hidden_channels=20, fine_blocks=2, coarse_blocks=3,
                       downsample_factor=4, use_control_relaxation=False,
                       use_conservative_flux=True, interface_transport_flux_gain=0.01)
