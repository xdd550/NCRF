from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..operators import NeumannResidualBlock, neumann_conv
from .common import FairComparatorConfig, _CommonDirectResidualForecaster


class _NeumannConvLSTMCell(nn.Module):
    """A conventional ConvLSTM cell using non-wrapping spatial kernels."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = neumann_conv(input_channels + hidden_channels, 4 * hidden_channels)

    def forward(self, value: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            hidden = value.new_zeros((value.shape[0], self.hidden_channels, *value.shape[-2:]))
            cell = torch.zeros_like(hidden)
        else:
            hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat((value, hidden), dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMForecaster(nn.Module):
    """True sequence-based control ConvLSTM, rolled out autoregressively.

    The ten observed ``(phi, u)`` pairs initialise the recurrent state in
    temporal order.  Every forecast step then consumes current predicted
    phase, prescribed next control, and the same causal progress scalar used
    by all learned comparators.
    """

    def __init__(self, config: FairComparatorConfig, *, hidden_channels: int = 60) -> None:
        super().__init__()
        self.config = config
        if hidden_channels < 16:
            raise ValueError("ConvLSTM hidden width is too small.")
        self.cell = _NeumannConvLSTMCell(3, hidden_channels)
        self.head = nn.Sequential(neumann_conv(hidden_channels, hidden_channels), nn.SiLU(), neumann_conv(hidden_channels, 1))

    def _validate(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> None:
        _CommonDirectResidualForecaster._validate(self, context_phi, context_u, future_u)

    def forward(self, context_phi: torch.Tensor, context_u: torch.Tensor, future_u: torch.Tensor) -> torch.Tensor:
        self._validate(context_phi, context_u, future_u)
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        zeros = torch.zeros_like(context_phi[:, 0])
        for step in range(self.config.input_steps):
            state = self.cell(torch.cat((context_phi[:, step], context_u[:, step], zeros), dim=1), state)
        value, result = context_phi[:, -1], []
        horizon = future_u.shape[1]
        for step in range(horizon):
            control = future_u[:, step]
            progress = torch.full_like(control, (step + 1) / horizon)
            state = self.cell(torch.cat((value, control, progress), dim=1), state)
            raw = self.head(state[0])
            if not bool(torch.isfinite(raw).all()):
                raise RuntimeError("ConvLSTM produced a non-finite residual.")
            value = (value.float() + self.config.max_local_step_delta * torch.tanh(raw.float())).clamp(-1.0, 1.0)
            result.append(value)
        return torch.stack(result, dim=1)

