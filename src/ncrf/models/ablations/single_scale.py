from __future__ import annotations
import copy
from dataclasses import replace
import torch
from torch import nn
from ..backbone import VCHV15Config
from ..ncrf import V31NCRF, core_config
from ..operators import zero_flux_divergence


def single_scale_config(*, size: int = 128) -> VCHV15Config:
    """Nine fine-scale blocks compensate for the removed coarse-path capacity."""
    return replace(core_config(size=size), fine_blocks=9)


class V31SingleScaleNCRF(V31NCRF):
    """Fine-scale-only NCRF with matched learned capacity.

    ``VCHV15Config`` retains a positive ``coarse_blocks`` field solely because
    the historical config validates it.  Every coarse-path module is removed
    below and the recurrent coarse-state slot is a non-learned zero tensor.
    """

    def __init__(self, config: VCHV15Config, *, checkpoint_steps: bool = True):
        if config.fine_blocks != 9:
            raise ValueError("Single-scale screen is registered with exactly nine fine residual blocks.")
        super().__init__(config, variant="full", checkpoint_steps=checkpoint_steps)
        width = config.hidden_channels

        # The original fine input concatenates [value, fine state, control,
        # upsampled coarse features].  Retain only the first three inputs.
        old_fine_input = self.fine_input[0]
        if not isinstance(old_fine_input, nn.Conv2d):
            raise TypeError("Unexpected fine-input convolution type.")
        new_fine_input = copy.deepcopy(old_fine_input)
        new_fine_input.in_channels = 3 * width
        new_fine_input.weight = nn.Parameter(old_fine_input.weight[:, : 3 * width].detach().clone())
        self.fine_input[0] = new_fine_input

        # The full global-mean head receives coarse and fine feature means.
        # Keep fine means plus control statistics and rollout progress only.
        old_mean = self.mean_drift_head[0]
        if not isinstance(old_mean, nn.Linear):
            raise TypeError("Unexpected global-mean head input type.")
        new_mean = nn.Linear(width + 3, width, bias=old_mean.bias is not None)
        with torch.no_grad():
            new_mean.weight.copy_(torch.cat((old_mean.weight[:, :width], old_mean.weight[:, 2 * width:]), dim=1))
            if old_mean.bias is not None and new_mean.bias is not None:
                new_mean.bias.copy_(old_mean.bias)
        self.mean_drift_head[0] = new_mean

        # These are actual removals, not zeroed learned modules.  Hence any
        # benefit cannot be attributed to retained coarse-path capacity.
        self.context_coarse = None
        self.coarse_input = None
        self.coarse_blocks = None
        self.coarse_delta = None
        self.coarse_feedback = None
        self.coarse_gate = None
        self.coarse_candidate = None

    def _initial_state(self, phi, past_u):
        stacked = torch.cat((phi.flatten(1, 2), past_u.flatten(1, 2)), dim=1)
        fine = self.context_fine(stacked)
        relaxation = self.relaxation_initializer(past_u.flatten(1, 2))
        # Compatibility slot for the inherited recurrent-forward signature;
        # it has no parameters and is not used by the next step.
        return fine, torch.zeros_like(fine), relaxation

    def _step_impl(self, value, previous_value, fine_hidden, coarse_hidden, relaxation,
                   control, previous_control, progress, reference_mass):
        if reference_mass is not None:
            raise ValueError("No mass projection is part of the V31 core protocol.")
        control_fine, _control_coarse, condition = self._control_features(control, progress)
        value_feature = self.value_encoder(value)
        fine = self.fine_input(torch.cat((value_feature, fine_hidden, control_fine), dim=1))
        for block in self.fine_blocks:
            fine = block(fine, condition)

        next_r = (
            self.relaxation_memory_coefficient * relaxation
            + self.relaxation_control_coefficient * 0.5 * (previous_control + control)
        )
        q_input = torch.cat((fine, self.relaxation_encoder(next_r)), dim=1)
        q = torch.tanh(self.flux_potential(q_input))
        mobility = self.flux_mobility(fine).to(q.dtype)
        branch = self.config.flux_residual_gain * zero_flux_divergence(q, mobility)
        local = self._zero_mean(self.fine_delta(fine) + torch.zeros_like(next_r) + branch)
        stats = torch.cat((
            fine.float().mean((-2, -1)),
            control.float().mean((-2, -1)),
            control.float().std((-2, -1), unbiased=False),
            torch.full((value.shape[0], 1), progress, dtype=torch.float32, device=value.device),
        ), dim=1)
        mean_delta = self.config.max_global_mean_step_delta * torch.tanh(
            self.mean_drift_head(stats)
        ).view(-1, 1, 1, 1)
        pre_bound = value.float() + self.config.max_local_step_delta * torch.tanh(local) + mean_delta
        prediction = pre_bound.clamp(-1, 1)
        with torch.no_grad():
            self.last_step_audit = {
                "branch_rms_last_step": float(branch.float().square().mean().sqrt()),
                "local_tanh_saturation_last_step": float((local.abs() > 3).float().mean()),
                "clip_fraction_last_step": float((pre_bound.abs() > 1).float().mean()),
            }
        next_fine = self._update_hidden(
            fine_hidden, fine, self.fine_feedback(prediction.to(value.dtype)), self.fine_gate, self.fine_candidate
        )
        next_coarse = torch.zeros_like(next_fine)
        zero = torch.zeros_like(prediction)
        scalar_zero = zero[..., :1, :1]
        return (prediction, next_fine, next_coarse, next_r, mean_delta, branch, zero,
                torch.ones_like(branch), zero, zero, zero, scalar_zero, zero, scalar_zero, scalar_zero)

