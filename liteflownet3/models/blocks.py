"""Building blocks for the LiteFlowNet3 PyTorch implementation.

These layers are inspired by the original LiteFlowNet family of optical
flow networks. They provide reusable components for the feature pyramid,
cost volume construction and cascaded flow estimation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def default_activation(inplace: bool = True) -> nn.Module:
    """Return the default activation function used throughout the model."""

    return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)


class ConvBlock(nn.Sequential):
    """A convenience wrapper around Conv2d + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        activation: bool = True,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = (kernel_size // 2) * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        ]

        if activation:
            layers.append(default_activation())

        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """A simple residual block consisting of two convolutional layers."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            ConvBlock(channels, hidden_channels, kernel_size=3),
            ConvBlock(hidden_channels, channels, kernel_size=3, activation=False),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - refer to base class.
        return default_activation()(x + self.block(x))


class FeaturePyramidExtractor(nn.Module):
    """Constructs the multi-scale feature pyramid used by LiteFlowNet3."""

    def __init__(self, channels: Iterable[int]) -> None:
        super().__init__()
        in_channels = 3
        self.layers = nn.ModuleList()

        for out_channels in channels:
            level = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=3, stride=2),
                ConvBlock(out_channels, out_channels, kernel_size=3),
                ConvBlock(out_channels, out_channels, kernel_size=3),
            )
            self.layers.append(level)
            in_channels = out_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        features: List[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


def warp(features: Tensor, flow: Tensor) -> Tensor:
    """Warp features according to the optical flow using grid sampling."""

    b, c, h, w = features.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=features.device, dtype=features.dtype),
        torch.linspace(-1.0, 1.0, w, device=features.device, dtype=features.dtype),
        indexing="ij",
    )

    base_grid = torch.stack((grid_x, grid_y), dim=-1)

    flow_norm = torch.zeros_like(flow)
    flow_norm[:, 0, :, :] = flow[:, 0, :, :] / ((w - 1.0) / 2.0)
    flow_norm[:, 1, :, :] = flow[:, 1, :, :] / ((h - 1.0) / 2.0)

    sampling_grid = base_grid + flow_norm.permute(0, 2, 3, 1)

    return F.grid_sample(features, sampling_grid, align_corners=True)


def build_cost_volume(
    ref: Tensor,
    target: Tensor,
    search_range: int,
    normalize: bool = True,
) -> Tensor:
    """Compute the correlation cost volume between reference and target features."""

    if search_range < 1:
        raise ValueError("search_range must be >= 1")

    b, c, h, w = ref.shape
    padded_target = F.pad(target, (search_range, search_range, search_range, search_range))

    cost_volumes: List[Tensor] = []
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            shifted = padded_target[:, :, search_range + dy : search_range + dy + h, search_range + dx : search_range + dx + w]
            cost = (ref * shifted).sum(1, keepdim=True)
            cost_volumes.append(cost)

    cost_volume = torch.cat(cost_volumes, dim=1)

    if normalize:
        cost_volume = cost_volume / torch.sqrt(ref.new_tensor(float(c)))

    return cost_volume


class FlowEstimatorDense(nn.Module):
    """Dense flow estimator similar to the cascaded modules in LiteFlowNet."""

    def __init__(self, in_channels: int, hidden_channels: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for hidden in hidden_channels:
            layers.append(ConvBlock(in_channels, hidden, kernel_size=3))
            in_channels = hidden
        layers.append(nn.Conv2d(in_channels, 2, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - refer to base class.
        return self.network(x)


class ContextNetwork(nn.Module):
    """Context refinement network refining the estimated flow."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(in_channels, 128, kernel_size=3),
            ConvBlock(128, 128, kernel_size=3, dilation=2, padding=2),
            ConvBlock(128, 128, kernel_size=3, dilation=4, padding=4),
            ConvBlock(128, 96, kernel_size=3, dilation=8, padding=8),
            ConvBlock(96, 64, kernel_size=3, dilation=16, padding=16),
            ConvBlock(64, 32, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - refer to base class.
        return self.network(x)


@dataclass
class DecoderConfig:
    """Configuration container for each decoder stage."""

    search_range: int = 4
    hidden_channels: Tuple[int, ...] = (128, 128, 96, 64, 32)

