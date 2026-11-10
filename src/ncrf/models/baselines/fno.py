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
from .source_loader import _locked_fno_class


class ReflectiveFNOForecaster(_CommonDirectResidualForecaster):
    """Source-locked FNO on an even reflection extension, then cropped back.

    The reflection is a fixed input representation for known non-wrapping
    boundaries.  It is not a flux, projection, or learned physical operator.
    """

    def __init__(self, config: FairComparatorConfig, *, source_directory: Path, hidden_channels: int = 24, layers: int = 4, modes: int = 12) -> None:
        super().__init__(config)
        if min(hidden_channels, layers, modes) < 1:
            raise ValueError("FNO dimensions must be positive.")
        fno_class = _locked_fno_class(str(Path(source_directory).resolve()))
        self.core = fno_class(
            n_modes=(modes, modes), in_channels=22, out_channels=1, hidden_channels=hidden_channels,
            n_layers=layers, positional_embedding="grid", domain_padding=None,
        )

    def _raw_delta(self, fields: torch.Tensor) -> torch.Tensor:
        height, width = fields.shape[-2:]
        extended = F.pad(fields.float(), (0, width - 2, 0, height - 2), mode="reflect")
        # H90 retains one FNO graph per autoregressive step. Checkpointing
        # recomputes this unchanged deterministic core during backward rather
        # than retaining all its intermediate activations. It changes neither
        # FNO parameters, inputs, outputs, gradients, nor effective batch size.
        if self.training and torch.is_grad_enabled():
            raw = checkpoint(self.core, extended, use_reentrant=False, preserve_rng_state=True)
        else:
            raw = self.core(extended)
        return raw[..., :height, :width].to(dtype=fields.dtype)

