"""LiteFlowNet3 implementation in PyTorch.

The implementation is a faithful PyTorch re-imagining of the original
LiteFlowNet3 architecture described in:

    Hui et al., "A Lightweight Optical Flow CNN - Revisiting Data Fidelity"

The design follows the coarse-to-fine cascade with descriptor matching
and flow regularisation. Some practical adaptations are made for
readability and PyTorch idioms while keeping the inference graph and
tensor operations close to the published design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import (
    ContextNetwork,
    DecoderConfig,
    FeaturePyramidExtractor,
    FlowEstimatorDense,
    build_cost_volume,
    warp,
)


def upsample_flow(flow: Tensor, scale_factor: float = 2.0) -> Tensor:
    """Bilinearly upsample an optical flow field preserving displacement."""

    return F.interpolate(flow, scale_factor=scale_factor, mode="bilinear", align_corners=True) * scale_factor


@dataclass
class LiteFlowNet3Config:
    """Configuration of LiteFlowNet3 architecture."""

    pyramid_channels: Sequence[int] = (32, 48, 64, 96, 128, 192)
    decoder_configs: Sequence[DecoderConfig] = field(
        default_factory=lambda: (
            DecoderConfig(search_range=4, hidden_channels=(128, 128, 96, 64, 32)),
            DecoderConfig(search_range=4, hidden_channels=(128, 128, 96, 64, 32)),
            DecoderConfig(search_range=4, hidden_channels=(128, 96, 64, 32, 32)),
            DecoderConfig(search_range=4, hidden_channels=(96, 64, 32, 32)),
            DecoderConfig(search_range=3, hidden_channels=(96, 64, 32, 32)),
        )
    )
    context_channels: int | None = None


class LiteFlowNet3(nn.Module):
    """LiteFlowNet3 network for two-frame optical flow estimation."""

    def __init__(self, config: LiteFlowNet3Config | None = None) -> None:
        super().__init__()
        if config is None:
            config = LiteFlowNet3Config()

        self.config = config
        self.feature_extractor = FeaturePyramidExtractor(config.pyramid_channels)

        decoder_in_channels: List[int] = []
        for pyramid_channels, decoder_cfg in zip(reversed(config.pyramid_channels[:-1]), config.decoder_configs):
            # cost volume size + warped features + previous flow
            search_area = (2 * decoder_cfg.search_range + 1) ** 2
            decoder_in_channels.append(search_area + pyramid_channels + 2)

        self.decoders = nn.ModuleList(
            [
                FlowEstimatorDense(in_channels, decoder_cfg.hidden_channels)
                for in_channels, decoder_cfg in zip(decoder_in_channels, config.decoder_configs)
            ]
        )

        context_in_channels = config.context_channels
        if context_in_channels is None:
            context_in_channels = decoder_in_channels[-1]

        self.context_net = ContextNetwork(context_in_channels)

    def forward(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """Estimate optical flow from ``frame1`` to ``frame2``.

        Parameters
        ----------
        frame1, frame2:
            Input RGB frames of shape ``(B, 3, H, W)`` normalised to ``[0, 1]``.
        """

        if frame1.shape != frame2.shape:
            raise ValueError("Input frames must have the same shape")

        features1 = self.feature_extractor(frame1)
        features2 = self.feature_extractor(frame2)

        flow: Tensor | None = None
        flows: List[Tensor] = []

        decoder_levels = zip(
            reversed(features1[:-1]),
            reversed(features2[:-1]),
            self.decoders,
            self.config.decoder_configs,
        )

        last_context_input: Tensor | None = None

        for feat1, feat2, decoder, decoder_cfg in decoder_levels:
            if flow is None:
                flow = torch.zeros(frame1.size(0), 2, feat1.size(2), feat1.size(3), device=frame1.device, dtype=frame1.dtype)
            else:
                flow = upsample_flow(flow)

            warped_feat2 = warp(feat2, flow)
            cost_volume = build_cost_volume(feat1, warped_feat2, search_range=decoder_cfg.search_range)
            decoder_input = torch.cat([cost_volume, feat1, flow], dim=1)
            flow_delta = decoder(decoder_input)
            flow = flow + flow_delta
            flows.append(flow)
            last_context_input = decoder_input

        if last_context_input is None:
            raise RuntimeError("Decoder did not run; check configuration")

        refined_flow = flow + self.context_net(last_context_input)

        return upsample_flow(refined_flow, scale_factor=2.0)


__all__ = ["LiteFlowNet3", "LiteFlowNet3Config"]

