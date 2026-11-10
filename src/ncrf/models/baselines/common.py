from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..operators import NeumannResidualBlock, neumann_conv


@dataclass(frozen=True)
class FairComparatorConfig:
    """Fixed common causal interface and generic numerical guard."""

    input_steps: int = 10
    image_size: int = 128
    max_local_step_delta: float = 0.16

    def __post_init__(self) -> None:
        if self.input_steps != 10:
            raise ValueError("The registered comparison protocol requires ten context frames.")
        if self.image_size < 16 or self.image_size % 8:
            raise ValueError("image_size must be divisible by eight and at least 16.")
        if not 0.0 < self.max_local_step_delta <= 0.25:
            raise ValueError("max_local_step_delta must be in (0, 0.25].")


class _CommonDirectResidualForecaster(nn.Module):
    """Common 22-field causal adapter and non-physical bounded rollout."""

    def __init__(self, config: FairComparatorConfig) -> None:
        super().__init__()
        self.config = config

    def _validate(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> None:
        shape = (self.config.input_steps, 1, self.config.image_size, self.config.image_size)
        field = (1, self.config.image_size, self.config.image_size)
        if context_phi.ndim != 5 or tuple(context_phi.shape[1:]) != shape:
            raise ValueError(f"Expected phase history [batch,{shape}], got {tuple(context_phi.shape)}.")
        if context_u.shape != context_phi.shape:
            raise ValueError("The control-history tensor must exactly match phase-history geometry.")
        if future_u.ndim != 5 or future_u.shape[0] != context_phi.shape[0] or tuple(future_u.shape[2:]) != field:
            raise ValueError(f"Expected prescribed future controls [batch,horizon,{field}], got {tuple(future_u.shape)}.")
        if future_u.shape[1] < 1 or not all(bool(torch.isfinite(item).all()) for item in (context_phi, context_u, future_u)):
            raise ValueError("All comparator inputs must be finite and contain at least one future-control step.")

    @staticmethod
    def _features(phi_history: torch.Tensor, u_history: torch.Tensor, prescribed_u: torch.Tensor, progress: float) -> torch.Tensor:
        progress_field = torch.full_like(prescribed_u, progress)
        fields = torch.cat(
            (phi_history.flatten(start_dim=1, end_dim=2), u_history.flatten(start_dim=1, end_dim=2), prescribed_u, progress_field),
            dim=1,
        )
        if fields.shape[1] != 22:
            raise RuntimeError("The common comparator adapter must yield 22 channels.")
        return fields

    def _raw_delta(self, fields: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract guard
        raise NotImplementedError

    def forward(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> torch.Tensor:
        self._validate(context_phi, context_u, future_u)
        phi_history, u_history = context_phi, context_u
        value = phi_history[:, -1]
        result: list[torch.Tensor] = []
        horizon = future_u.shape[1]
        for step in range(horizon):
            control = future_u[:, step]
            raw = self._raw_delta(self._features(phi_history, u_history, control, (step + 1) / horizon))
            if raw.shape != value.shape or not bool(torch.isfinite(raw).all()):
                raise RuntimeError("Comparator core returned a non-finite or incorrectly shaped residual.")
            value = (value.float() + self.config.max_local_step_delta * torch.tanh(raw.float())).clamp(-1.0, 1.0)
            result.append(value)
            phi_history = torch.cat((phi_history[:, 1:], value.unsqueeze(1).to(dtype=phi_history.dtype)), dim=1)
            u_history = torch.cat((u_history[:, 1:], control.unsqueeze(1).to(dtype=u_history.dtype)), dim=1)
        return torch.stack(result, dim=1)

