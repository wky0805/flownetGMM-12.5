"""Modulation modules for LiteFlowNet3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


def _default_pooling(kernel_size: int) -> nn.Module:
    padding = kernel_size // 2
    return nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)


@dataclass
class DualGateConfig:
    """Configuration for the dual-gated modulation module."""

    smoothing_kernel: int = 3
    min_confidence: float = 1e-4


class DualGateModulator(nn.Module):
    """Applies dual-channel gating to modulate flow refinements."""

    def __init__(self, config: DualGateConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DualGateConfig()

        self.config = config
        self.smoother = _default_pooling(config.smoothing_kernel)

    def forward(
        self,
        flow_delta: Tensor,
        confidence_logits: Tensor,
        outlier_logits: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Blend the flow update using confidence and outlier probability maps."""

        confidence = torch.sigmoid(confidence_logits)
        confidence = torch.clamp(confidence, min=self.config.min_confidence, max=1.0)
        outlier_prob = torch.sigmoid(outlier_logits)

        smoothed_delta = self.smoother(flow_delta)

        base_weight = confidence + (1.0 - confidence) * (1.0 - outlier_prob)
        smooth_weight = (1.0 - confidence) * outlier_prob

        modulated_delta = flow_delta * base_weight + smoothed_delta * smooth_weight
        return modulated_delta, confidence, outlier_prob


__all__ = ["DualGateModulator", "DualGateConfig"]
