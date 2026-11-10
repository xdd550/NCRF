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


class _NeumannDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            neumann_conv(in_channels, out_channels), nn.ReLU(inplace=True),
            neumann_conv(out_channels, out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class UNetForecaster(_CommonDirectResidualForecaster):
    """Standard compact U-Net, with only boundary padding aligned to the data."""

    def __init__(self, config: FairComparatorConfig, *, base_channels: int = 10, levels: int = 3) -> None:
        super().__init__(config)
        if levels != 3 or base_channels < 4:
            raise ValueError("The registered U-Net has three levels and a positive base width.")
        widths = [base_channels * (2**index) for index in range(levels + 1)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous = 22
        for width in widths[:-1]:
            self.encoders.append(_NeumannDoubleConv(previous, width))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            previous = width
        self.bottleneck = _NeumannDoubleConv(widths[-2], widths[-1])
        self.decoders = nn.ModuleList()
        for index in range(levels - 1, -1, -1):
            self.decoders.append(_NeumannDoubleConv(widths[index + 1] + widths[index], widths[index]))
        self.head = neumann_conv(widths[0], 1, kernel_size=1)

    def _raw_delta(self, fields: torch.Tensor) -> torch.Tensor:
        value, skips = fields, []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            value = encoder(value)
            skips.append(value)
            value = pool(value)
        value = self.bottleneck(value)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat((skip, value), dim=1))
        return self.head(value)

