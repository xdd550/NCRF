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


class PlainResCNNForecaster(_CommonDirectResidualForecaster):
    """Single-scale residual CNN with no NCRF-specific mechanism."""

    def __init__(self, config: FairComparatorConfig, *, width: int = 56, blocks: int = 3) -> None:
        super().__init__(config)
        if width < 16 or blocks < 1:
            raise ValueError("Plain ResCNN width/blocks are invalid.")
        self.stem = nn.Sequential(neumann_conv(22, width), nn.SiLU())
        self.blocks = nn.Sequential(*(NeumannResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Sequential(neumann_conv(width, width), nn.SiLU(), neumann_conv(width, 1))

    def _raw_delta(self, fields: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(fields)))

