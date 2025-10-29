"""Adversarial discriminator modules for LiteFlowNet3 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


@dataclass
class PatchDiscriminatorConfig:
    """Configuration for the PatchGAN-style discriminator."""

    in_channels: int = 1
    base_channels: int = 32
    max_channels: int = 256
    num_layers: int = 4
    use_spectral_norm: bool = True


class PatchDiscriminator(nn.Module):
    """Patch-based discriminator that classifies residual maps as normal/anomalous."""

    def __init__(self, config: PatchDiscriminatorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PatchDiscriminatorConfig()

        if config.num_layers < 2:
            raise ValueError("PatchDiscriminator requires at least two layers")

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        out_channels = config.base_channels

        for i in range(config.num_layers - 1):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if config.use_spectral_norm:
                conv = spectral_norm(conv)
            layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
            in_channels = out_channels
            out_channels = min(out_channels * 2, config.max_channels)

        conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        if config.use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)

        self.model = nn.Sequential(*layers)

    def forward(self, residual_map: Tensor) -> Tensor:
        """Return a logit map whose values indicate the likelihood of anomalies."""

        if residual_map.dim() != 4:
            raise ValueError("Residual map must be a 4D tensor (N, C, H, W)")
        return self.model(residual_map)

    def predict_proba(self, residual_map: Tensor) -> Tensor:
        """Return probabilities by applying a sigmoid to the discriminator logits."""

        return torch.sigmoid(self.forward(residual_map))


__all__ = ["PatchDiscriminator", "PatchDiscriminatorConfig"]
