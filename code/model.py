"""Compact residual 4x super-resolution network for four Sentinel-2 bands."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class CompactSR4x(nn.Module):
    """Predicts a learned residual on top of bicubic upsampling."""

    def __init__(self, bands: int = 4, channels: int = 32, blocks: int = 6) -> None:
        super().__init__()
        self.head = nn.Conv2d(bands, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, bands, 3, padding=1),
        )
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
        features = self.head(x)
        residual = self.tail(self.body(features))
        return torch.clamp(baseline + residual, 0.0, 1.0)
