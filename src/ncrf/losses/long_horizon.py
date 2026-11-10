"""Long-horizon VCH loss whose spatial terms respect zero-flux boundaries."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class VCHV15NeumannLongHorizonLoss(nn.Module):
    """V11-compatible objective with non-wrapping spatial derivatives.

    Field, coarse, late-frame and soft-mean terms retain the registered V11
    definitions so N0--N3 and V15 external baselines share one objective. The
    gradient/interface terms use forward interior differences and set the
    outer normal derivative to zero, matching the VCH-v2 Neumann geometry.
    """

    def __init__(
        self,
        *,
        gradient_weight: float = 0.10,
        interface_weight: float = 0.08,
        coarse_weight: float = 0.35,
        mean_weight: float = 0.20,
        boundary_weight: float = 0.0,
        boundary_match_weight: float = 0.0,
        adaptive_boundary_match_max_weight: float = 0.0,
        adaptive_boundary_match_tolerance: float = 0.0,
        adaptive_boundary_match_temperature: float = 0.001,
        adaptive_boundary_match_loss_cap: float = 0.0,
        adaptive_late_interface_max_weight: float = 0.0,
        adaptive_late_interface_tolerance: float = 0.0,
        adaptive_late_interface_temperature: float = 0.001,
        adaptive_late_interface_loss_cap: float = 0.0,
        adaptive_late_interface_band: float = 0.20,
        adaptive_late_interface_start_fraction: float = 1.0 / 3.0,
        adaptive_late_interface_min_horizon: int = 50,
        fixed_interface_morphology_max_weight: float = 0.0,
        interface_morphology_tv_weight: float = 0.25,
        interface_morphology_band: float = 0.20,
        interface_morphology_start_fraction: float = 1.0 / 3.0,
        interface_morphology_min_horizon: int = 50,
        interface_morphology_loss_cap: float = 0.0,
        use_adaptive_interface_morphology_activity: bool = False,
        interface_morphology_activity_min: float = 0.50,
        interface_morphology_activity_max: float = 2.00,
        interface_registration_max_weight: float = 0.0,
        interface_registration_band: float = 0.20,
        interface_registration_band_l1_weight: float = 0.20,
        interface_registration_start_fraction: float = 1.0 / 3.0,
        interface_registration_min_horizon: int = 50,
        interface_registration_loss_cap: float = 0.0,
        mean_tolerance: float = 0.002,
        late_frame_weight: float = 2.5,
        coarse_pool: int = 4,
    ) -> None:
        super().__init__()
        if min(
            gradient_weight,
            interface_weight,
            coarse_weight,
            mean_weight,
            boundary_weight,
            boundary_match_weight,
            adaptive_boundary_match_max_weight,
            adaptive_boundary_match_tolerance,
            adaptive_boundary_match_loss_cap,
            adaptive_late_interface_max_weight,
            adaptive_late_interface_tolerance,
            adaptive_late_interface_loss_cap,
            fixed_interface_morphology_max_weight,
            interface_morphology_tv_weight,
            interface_morphology_loss_cap,
            interface_registration_max_weight,
            interface_registration_band_l1_weight,
            interface_registration_loss_cap,
        ) < 0:
            raise ValueError("Loss weights must be non-negative.")
        if mean_tolerance < 0 or late_frame_weight < 1.0 or coarse_pool < 2:
            raise ValueError("Invalid V15 long-horizon loss settings.")
        if not 0.0 <= adaptive_boundary_match_loss_cap <= 1.0:
            raise ValueError("adaptive_boundary_match_loss_cap must be in [0, 1].")
        if adaptive_boundary_match_max_weight > 0.0 and (
            adaptive_boundary_match_temperature <= 0.0 or adaptive_boundary_match_loss_cap <= 0.0
        ):
            raise ValueError("Adaptive boundary matching requires a positive temperature and loss cap.")
        if not 0.0 < adaptive_late_interface_band <= 1.0:
            raise ValueError("adaptive_late_interface_band must lie in (0, 1].")
        if not 0.0 <= adaptive_late_interface_start_fraction < 1.0:
            raise ValueError("adaptive_late_interface_start_fraction must lie in [0, 1).")
        if adaptive_late_interface_min_horizon < 1:
            raise ValueError("adaptive_late_interface_min_horizon must be positive.")
        if adaptive_late_interface_max_weight > 0.0 and (
            adaptive_late_interface_temperature <= 0.0 or adaptive_late_interface_loss_cap <= 0.0
        ):
            raise ValueError("Adaptive late-interface calibration requires a positive temperature and loss cap.")
        if not 0.0 < interface_morphology_band <= 1.0:
            raise ValueError("interface_morphology_band must lie in (0, 1].")
        if not 0.0 <= interface_morphology_start_fraction < 1.0:
            raise ValueError("interface_morphology_start_fraction must lie in [0, 1).")
        if interface_morphology_min_horizon < 1:
            raise ValueError("interface_morphology_min_horizon must be positive.")
        if not 0.0 <= interface_morphology_loss_cap <= 1.0:
            raise ValueError("interface_morphology_loss_cap must be in [0, 1].")
        if fixed_interface_morphology_max_weight > 0.0 and interface_morphology_loss_cap <= 0.0:
            raise ValueError("Fixed interface-morphology calibration requires a positive loss cap.")
        if use_adaptive_interface_morphology_activity and fixed_interface_morphology_max_weight <= 0.0:
            raise ValueError("Adaptive interface activity requires the fixed interface-morphology term.")
        if not 0.0 < interface_morphology_activity_min <= interface_morphology_activity_max <= 5.0:
            raise ValueError("Interface-morphology activity bounds must satisfy 0 < min <= max <= 5.")
        if not 0.0 < interface_registration_band <= 1.0:
            raise ValueError("interface_registration_band must lie in (0, 1].")
        if not 0.0 <= interface_registration_start_fraction < 1.0:
            raise ValueError("interface_registration_start_fraction must lie in [0, 1).")
        if interface_registration_min_horizon < 1:
            raise ValueError("interface_registration_min_horizon must be positive.")
        if not 0.0 <= interface_registration_loss_cap <= 1.0:
            raise ValueError("interface_registration_loss_cap must be in [0, 1].")
        if interface_registration_max_weight > 0.0 and interface_registration_loss_cap <= 0.0:
            raise ValueError("Interface registration requires a positive loss cap.")
        if boundary_match_weight > 0.0 and adaptive_boundary_match_max_weight > 0.0:
            raise ValueError("Use either fixed or adaptive boundary matching, not both.")
        self.gradient_weight = gradient_weight
        self.interface_weight = interface_weight
        self.coarse_weight = coarse_weight
        self.mean_weight = mean_weight
        self.boundary_weight = boundary_weight
        self.boundary_match_weight = boundary_match_weight
        self.adaptive_boundary_match_max_weight = adaptive_boundary_match_max_weight
        self.adaptive_boundary_match_tolerance = adaptive_boundary_match_tolerance
        self.adaptive_boundary_match_temperature = adaptive_boundary_match_temperature
        self.adaptive_boundary_match_loss_cap = adaptive_boundary_match_loss_cap
        self.adaptive_late_interface_max_weight = adaptive_late_interface_max_weight
        self.adaptive_late_interface_tolerance = adaptive_late_interface_tolerance
        self.adaptive_late_interface_temperature = adaptive_late_interface_temperature
        self.adaptive_late_interface_loss_cap = adaptive_late_interface_loss_cap
        self.adaptive_late_interface_band = adaptive_late_interface_band
        self.adaptive_late_interface_start_fraction = adaptive_late_interface_start_fraction
        self.adaptive_late_interface_min_horizon = adaptive_late_interface_min_horizon
        self.fixed_interface_morphology_max_weight = fixed_interface_morphology_max_weight
        self.interface_morphology_tv_weight = interface_morphology_tv_weight
        self.interface_morphology_band = interface_morphology_band
        self.interface_morphology_start_fraction = interface_morphology_start_fraction
        self.interface_morphology_min_horizon = interface_morphology_min_horizon
        self.interface_morphology_loss_cap = interface_morphology_loss_cap
        self.use_adaptive_interface_morphology_activity = use_adaptive_interface_morphology_activity
        self.interface_morphology_activity_min = interface_morphology_activity_min
        self.interface_morphology_activity_max = interface_morphology_activity_max
        self.interface_registration_max_weight = interface_registration_max_weight
        self.interface_registration_band = interface_registration_band
        self.interface_registration_band_l1_weight = interface_registration_band_l1_weight
        self.interface_registration_start_fraction = interface_registration_start_fraction
        self.interface_registration_min_horizon = interface_registration_min_horizon
        self.interface_registration_loss_cap = interface_registration_loss_cap
        self.mean_tolerance = mean_tolerance
        self.late_frame_weight = late_frame_weight
        self.coarse_pool = coarse_pool

    @staticmethod
    def _gradient(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward differences with no cross-boundary wraparound."""

        dx = torch.zeros_like(field)
        dy = torch.zeros_like(field)
        dx[..., :-1] = field[..., 1:] - field[..., :-1]
        dy[..., :-1, :] = field[..., 1:, :] - field[..., :-1, :]
        return dx, dy

    def _time_weights(self, value: torch.Tensor) -> torch.Tensor:
        steps = value.shape[1]
        weights = torch.linspace(1.0, self.late_frame_weight, steps, dtype=value.dtype, device=value.device)
        return (weights / weights.mean()).view(1, steps, 1, 1, 1)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
        return (value * time_weights).mean()

    @staticmethod
    def _outward_normal_differences(field: torch.Tensor) -> torch.Tensor:
        """Return one-sided outward normal finite differences on all edges."""

        top = field[..., 0, :] - field[..., 1, :]
        bottom = field[..., -1, :] - field[..., -2, :]
        left = field[..., :, 0] - field[..., :, 1]
        right = field[..., :, -1] - field[..., :, -2]
        return torch.cat((top.flatten(3), bottom.flatten(3), left.flatten(3), right.flatten(3)), dim=3)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 1:
            raise ValueError("V15 prediction and target must match [batch,time,1,H,W].")
        if prediction.shape[-1] % self.coarse_pool != 0 or prediction.shape[-2] % self.coarse_pool != 0:
            raise ValueError("Spatial dimensions must be divisible by coarse_pool.")

        time_weights = self._time_weights(prediction)
        field = self._weighted_mean((prediction - target).square(), time_weights)
        pred_x, pred_y = self._gradient(prediction)
        true_x, true_y = self._gradient(target)
        gradient = self._weighted_mean((pred_x - true_x).abs() + (pred_y - true_y).abs(), time_weights)

        interface_strength = torch.sqrt(true_x.square() + true_y.square() + 1e-8)
        interface_map = 1.0 + 0.50 * torch.tanh(interface_strength / 0.04)
        interface = self._weighted_mean(interface_map * (prediction - target).abs(), time_weights)

        shape = prediction.shape
        coarse_prediction = functional.avg_pool2d(prediction.flatten(0, 1), self.coarse_pool, self.coarse_pool).view(
            shape[0], shape[1], 1, shape[-2] // self.coarse_pool, shape[-1] // self.coarse_pool
        )
        coarse_target = functional.avg_pool2d(target.flatten(0, 1), self.coarse_pool, self.coarse_pool).view_as(coarse_prediction)
        coarse = self._weighted_mean((coarse_prediction - coarse_target).square(), time_weights)

        mean_error = (prediction.mean(dim=(-2, -1)) - target.mean(dim=(-2, -1))).abs()
        mean_hinge = self._weighted_mean(torch.relu(mean_error - self.mean_tolerance).square(), time_weights)
        prediction_boundary = self._outward_normal_differences(prediction)
        target_boundary = self._outward_normal_differences(target)
        boundary_per_frame = prediction_boundary.abs().mean(dim=(2, 3))
        boundary_l1 = (boundary_per_frame * time_weights[:, :, 0, 0, 0]).mean()
        boundary_match_per_frame = (prediction_boundary - target_boundary).abs().mean(dim=(2, 3))
        boundary_match_l1 = (boundary_match_per_frame * time_weights[:, :, 0, 0, 0]).mean()
        late_start = max(0, (3 * prediction.shape[1]) // 4)
        late_field_mse = (prediction[:, late_start:] - target[:, late_start:]).square().mean()
        late_boundary_match_l1 = (prediction_boundary[:, late_start:] - target_boundary[:, late_start:]).abs().mean()

        # The frozen-N2 H90 diagnosis shows that the dominant accumulated
        # error is on the phase interface and grows sharply after roughly one
        # third of the rollout.  This label-only training term stays off for
        # H10/H20 and early frames, then raises attention only when the
        # detached late-interface residual exceeds its declared tolerance. It
        # is capped relative to late field MSE so it cannot dominate the
        # registered objective.
        time_progress = torch.linspace(
            1.0 / prediction.shape[1], 1.0, prediction.shape[1], dtype=prediction.dtype, device=prediction.device
        ).view(1, prediction.shape[1], 1, 1, 1)
        interface_ramp = (
            (time_progress - self.adaptive_late_interface_start_fraction)
            / (1.0 - self.adaptive_late_interface_start_fraction)
        ).clamp(0.0, 1.0)
        target_interface = (target.abs() <= self.adaptive_late_interface_band).to(dtype=prediction.dtype)
        late_interface_mask = target_interface * interface_ramp
        late_interface_count = late_interface_mask.sum().clamp_min(torch.finfo(prediction.dtype).eps)
        late_interface_l1 = ((prediction - target).abs() * late_interface_mask).sum() / late_interface_count
        if (
            self.adaptive_late_interface_max_weight > 0.0
            and prediction.shape[1] >= self.adaptive_late_interface_min_horizon
        ):
            detached_interface = late_interface_l1.detach()
            interface_excess = (detached_interface - self.adaptive_late_interface_tolerance).clamp_min(0.0)
            adaptive_late_interface_gate = interface_excess / (
                interface_excess + self.adaptive_late_interface_temperature
            )
            uncapped_interface_weight = adaptive_late_interface_gate * self.adaptive_late_interface_max_weight
            capped_interface_weight = (
                self.adaptive_late_interface_loss_cap
                * late_field_mse.detach()
                / detached_interface.clamp_min(torch.finfo(prediction.dtype).eps)
            )
            adaptive_late_interface_weight = torch.minimum(uncapped_interface_weight, capped_interface_weight)
            adaptive_late_interface_term = adaptive_late_interface_weight * late_interface_l1
            adaptive_late_interface_cap = self.adaptive_late_interface_loss_cap * late_field_mse.detach()
        else:
            adaptive_late_interface_gate = prediction.new_zeros(())
            adaptive_late_interface_weight = prediction.new_zeros(())
            adaptive_late_interface_term = prediction.new_zeros(())
            adaptive_late_interface_cap = prediction.new_zeros(())

        # V16 M2/M3: a training-only interface-morphology term.  M2 uses a
        # fixed late-frame weighting; M3 replaces only that weighting with a
        # causal target-side activity score s_t - s_(t-1).  Neither variant
        # adds a model parameter, reads a future target at inference, or
        # changes the zero-flux N2 update.  Its detached cap keeps the term a
        # bounded fraction of late field MSE so morphology cannot overwhelm
        # data fidelity during autoregressive training.
        morphology_progress = torch.linspace(
            1.0 / prediction.shape[1], 1.0, prediction.shape[1], dtype=prediction.dtype, device=prediction.device
        ).view(1, prediction.shape[1])
        morphology_ramp = (
            (morphology_progress - self.interface_morphology_start_fraction)
            / (1.0 - self.interface_morphology_start_fraction)
        ).clamp(0.0, 1.0)
        soft_prediction = torch.exp(-torch.square(prediction / self.interface_morphology_band))
        soft_target = torch.exp(-torch.square(target / self.interface_morphology_band))
        soft_overlap_per_frame = torch.square(soft_prediction - soft_target).mean(dim=(2, 3, 4))
        pred_grad_x, pred_grad_y = self._gradient(prediction)
        target_grad_x, target_grad_y = self._gradient(target)
        prediction_tv = torch.sqrt(pred_grad_x.square() + pred_grad_y.square() + 1e-8).mean(dim=(2, 3, 4))
        target_tv = torch.sqrt(target_grad_x.square() + target_grad_y.square() + 1e-8).mean(dim=(2, 3, 4))
        tv_relative_error_per_frame = (prediction_tv - target_tv).abs() / target_tv.detach().clamp_min(1e-4)
        morphology_per_frame = soft_overlap_per_frame + self.interface_morphology_tv_weight * tv_relative_error_per_frame
        activity = torch.zeros_like(morphology_per_frame)
        if prediction.shape[1] > 1:
            activity[:, 1:] = (soft_target[:, 1:] - soft_target[:, :-1]).abs().mean(dim=(2, 3, 4))
        eligible = morphology_ramp.expand_as(activity)
        eligible_mass = eligible.sum(dim=1, keepdim=True).clamp_min(torch.finfo(prediction.dtype).eps)
        activity_mean = (activity * eligible).sum(dim=1, keepdim=True) / eligible_mass
        activity_weight = (activity / activity_mean.clamp_min(torch.finfo(prediction.dtype).eps)).clamp(
            self.interface_morphology_activity_min, self.interface_morphology_activity_max
        )
        if not self.use_adaptive_interface_morphology_activity:
            activity_weight = torch.ones_like(activity_weight)
        morphology_weight = eligible * activity_weight
        morphology_count = morphology_weight.sum().clamp_min(torch.finfo(prediction.dtype).eps)
        morphology_overlap = (soft_overlap_per_frame * morphology_weight).sum() / morphology_count
        morphology_tv = (tv_relative_error_per_frame * morphology_weight).sum() / morphology_count
        morphology_raw = (morphology_per_frame * morphology_weight).sum() / morphology_count
        if (
            self.fixed_interface_morphology_max_weight > 0.0
            and prediction.shape[1] >= self.interface_morphology_min_horizon
        ):
            detached_morphology = morphology_raw.detach()
            morphology_cap = self.interface_morphology_loss_cap * late_field_mse.detach()
            cap_weight = morphology_cap / detached_morphology.clamp_min(torch.finfo(prediction.dtype).eps)
            morphology_weight_effective = torch.minimum(
                prediction.new_tensor(self.fixed_interface_morphology_max_weight), cap_weight
            )
            interface_morphology_term = morphology_weight_effective * morphology_raw
        else:
            morphology_cap = prediction.new_zeros(())
            morphology_weight_effective = prediction.new_zeros(())
            interface_morphology_term = prediction.new_zeros(())

        # V22 R1: direct late-frame interface registration.  The V21 B2
        # multiresolution backbone sharply reduces field RMSE but its H90
        # interface Dice falls despite a near-correct interface area.  That
        # signature is a position/registration error, not missing detail:
        # its total variation is already too high.  Unlike V16 morphology,
        # this term therefore has no TV component.  It compares smooth,
        # differentiable near-zero bands by a soft Dice plus a target-band
        # L1 term, and is capped against late field MSE.  It is label-only
        # during training and has no model parameter or inference input.
        registration_progress = torch.linspace(
            1.0 / prediction.shape[1], 1.0, prediction.shape[1], dtype=prediction.dtype, device=prediction.device
        ).view(1, prediction.shape[1])
        registration_ramp = (
            (registration_progress - self.interface_registration_start_fraction)
            / (1.0 - self.interface_registration_start_fraction)
        ).clamp(0.0, 1.0)
        registration_prediction = torch.exp(-torch.square(prediction / self.interface_registration_band))
        registration_target = torch.exp(-torch.square(target / self.interface_registration_band))
        registration_numerator = 2.0 * (registration_prediction * registration_target).sum(dim=(2, 3, 4)) + 1e-6
        registration_denominator = (
            registration_prediction.square().sum(dim=(2, 3, 4))
            + registration_target.square().sum(dim=(2, 3, 4))
            + 1e-6
        )
        registration_dice_loss = 1.0 - registration_numerator / registration_denominator
        registration_target_count = registration_target.sum(dim=(2, 3, 4)).clamp_min(torch.finfo(prediction.dtype).eps)
        registration_band_l1 = (
            registration_target * (prediction - target).abs()
        ).sum(dim=(2, 3, 4)) / registration_target_count
        registration_per_frame = registration_dice_loss + self.interface_registration_band_l1_weight * registration_band_l1
        registration_mass = registration_ramp.sum().clamp_min(torch.finfo(prediction.dtype).eps)
        registration_raw = (registration_per_frame * registration_ramp).sum() / (prediction.shape[0] * registration_mass)
        registration_dice_metric = (registration_dice_loss * registration_ramp).sum().detach() / (
            prediction.shape[0] * registration_mass
        )
        registration_band_l1_metric = (registration_band_l1 * registration_ramp).sum().detach() / (
            prediction.shape[0] * registration_mass
        )
        if (
            self.interface_registration_max_weight > 0.0
            and prediction.shape[1] >= self.interface_registration_min_horizon
        ):
            detached_registration = registration_raw.detach()
            registration_cap = self.interface_registration_loss_cap * late_field_mse.detach()
            registration_cap_weight = registration_cap / detached_registration.clamp_min(torch.finfo(prediction.dtype).eps)
            registration_weight = torch.minimum(prediction.new_tensor(self.interface_registration_max_weight), registration_cap_weight)
            interface_registration_term = registration_weight * registration_raw
        else:
            registration_cap = prediction.new_zeros(())
            registration_weight = prediction.new_zeros(())
            interface_registration_term = prediction.new_zeros(())

        # The rejected fixed 0.10 boundary-match loss altered every forecasting
        # step and increased long-horizon seed variance.  This conservative
        # alternative acts only on the final quarter: it is inactive below a
        # tolerance, rises smoothly with excess boundary mismatch, and its
        # detached coefficient caps its contribution relative to late field MSE.
        # Detaching the coefficient prevents the model from reducing the loss by
        # manipulating the gate or cap instead of reducing the boundary error.
        if self.adaptive_boundary_match_max_weight > 0.0:
            detached_match = late_boundary_match_l1.detach()
            excess = (detached_match - self.adaptive_boundary_match_tolerance).clamp_min(0.0)
            adaptive_gate = excess / (excess + self.adaptive_boundary_match_temperature)
            uncapped_weight = adaptive_gate * self.adaptive_boundary_match_max_weight
            capped_weight = (
                self.adaptive_boundary_match_loss_cap * late_field_mse.detach() / detached_match.clamp_min(torch.finfo(prediction.dtype).eps)
            )
            adaptive_weight = torch.minimum(uncapped_weight, capped_weight)
            adaptive_boundary_match_term = adaptive_weight * late_boundary_match_l1
            adaptive_boundary_match_cap = self.adaptive_boundary_match_loss_cap * late_field_mse.detach()
        else:
            adaptive_gate = prediction.new_zeros(())
            adaptive_weight = prediction.new_zeros(())
            adaptive_boundary_match_term = prediction.new_zeros(())
            adaptive_boundary_match_cap = prediction.new_zeros(())
        total = (
            field
            + self.gradient_weight * gradient
            + self.interface_weight * interface
            + self.coarse_weight * coarse
            + self.mean_weight * mean_hinge
            + self.boundary_weight * boundary_l1
            + self.boundary_match_weight * boundary_match_l1
            + adaptive_boundary_match_term
            + adaptive_late_interface_term
            + interface_morphology_term
            + interface_registration_term
        )
        return total, {
            "field_mse": field.detach(),
            "gradient_l1": gradient.detach(),
            "interface_l1": interface.detach(),
            "coarse_mse": coarse.detach(),
            "mean_hinge": mean_hinge.detach(),
            "mean_abs_error": self._weighted_mean(mean_error, time_weights).detach(),
            "boundary_zero_flux_l1": boundary_l1.detach(),
            "boundary_normal_match_l1": boundary_match_l1.detach(),
            "late_field_mse": late_field_mse.detach(),
            "late_boundary_normal_match_l1": late_boundary_match_l1.detach(),
            "adaptive_boundary_match_gate": adaptive_gate.detach(),
            "adaptive_boundary_match_weight": adaptive_weight.detach(),
            "adaptive_boundary_match_term": adaptive_boundary_match_term.detach(),
            "adaptive_boundary_match_cap": adaptive_boundary_match_cap.detach(),
            "late_interface_l1_abs_phi_le_band": late_interface_l1.detach(),
            "adaptive_late_interface_gate": adaptive_late_interface_gate.detach(),
            "adaptive_late_interface_weight": adaptive_late_interface_weight.detach(),
            "adaptive_late_interface_term": adaptive_late_interface_term.detach(),
            "adaptive_late_interface_cap": adaptive_late_interface_cap.detach(),
            "interface_morphology_soft_overlap": morphology_overlap.detach(),
            "interface_morphology_tv_relative_error": morphology_tv.detach(),
            "interface_morphology_raw": morphology_raw.detach(),
            "interface_morphology_activity_mean": activity_mean.mean().detach(),
            "interface_morphology_activity_weight_mean": activity_weight.mean().detach(),
            "interface_morphology_activity_weight_max": activity_weight.max().detach(),
            "interface_morphology_weight": morphology_weight_effective.detach(),
            "interface_morphology_term": interface_morphology_term.detach(),
            "interface_morphology_cap": morphology_cap.detach(),
            "interface_registration_soft_dice_loss": registration_dice_metric,
            "interface_registration_band_l1": registration_band_l1_metric,
            "interface_registration_raw": registration_raw.detach(),
            "interface_registration_weight": registration_weight.detach(),
            "interface_registration_term": interface_registration_term.detach(),
            "interface_registration_cap": registration_cap.detach(),
        }
