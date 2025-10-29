"""Gaussian mixture residual modelling for LiteFlowNet3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GMMConfig:
    """Configuration for the residual Gaussian mixture model."""

    num_components: int = 2
    em_momentum: float = 0.05
    min_variance: float = 1e-4
    sparse_lambda: float = 1e-3
    anomaly_component: int = 1
    eps: float = 1e-6


class ResidualGMM(nn.Module):
    """Mixture of Gaussians modelling the distribution of flow residuals."""

    def __init__(self, config: GMMConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GMMConfig()

        if config.num_components < 2:
            raise ValueError("ResidualGMM requires at least two mixture components")
        if not (0 <= config.anomaly_component < config.num_components):
            raise ValueError("Anomaly component index must be within the number of components")

        self.config = config

        self.logits = nn.Parameter(torch.zeros(config.num_components))
        self.means = nn.Parameter(torch.zeros(config.num_components))
        self.log_vars = nn.Parameter(torch.zeros(config.num_components))

    @property
    def mixture_weights(self) -> Tensor:
        return F.softmax(self.logits, dim=0)

    @property
    def variances(self) -> Tensor:
        return torch.exp(self.log_vars).clamp_min(self.config.min_variance)

    def forward(self, residual: Tensor, update_em: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute negative log likelihood of residuals under the mixture model."""

        if residual.dim() < 2:
            raise ValueError("Residual tensor is expected to have flow channel dimension")

        # Use magnitude of the residual flow vector for probabilistic modelling
        residual_magnitude = torch.norm(residual, dim=1, keepdim=True)
        flattened = residual_magnitude.reshape(-1)

        if flattened.numel() == 0:
            raise RuntimeError("Residual tensor is empty; cannot compute GMM loss")

        x = flattened.unsqueeze(-1)
        variances = self.variances
        log_weights = F.log_softmax(self.logits, dim=0)

        log_component_prob = -0.5 * (
            ((x - self.means) ** 2) / variances + torch.log(2 * torch.pi * variances)
        )
        log_joint = log_component_prob + log_weights
        log_likelihood = torch.logsumexp(log_joint, dim=1)
        nll = -log_likelihood.mean()

        weights = self.mixture_weights
        sparse_penalty = self.config.sparse_lambda * weights[self.config.anomaly_component]
        loss = nll + sparse_penalty

        responsibilities = torch.softmax(log_joint, dim=1)

        if self.training and update_em:
            with torch.no_grad():
                Nk = responsibilities.sum(dim=0) + self.config.eps
                inv_Nk = 1.0 / Nk
                expected_means = (responsibilities * x).sum(dim=0) * inv_Nk
                expected_vars = (responsibilities * (x - expected_means) ** 2).sum(dim=0) * inv_Nk
                expected_logits = torch.log(Nk / Nk.sum())

                self.means.data.lerp_(expected_means, self.config.em_momentum)
                updated_vars = expected_vars.clamp_min(self.config.min_variance)
                self.log_vars.data.lerp_(torch.log(updated_vars), self.config.em_momentum)
                self.logits.data.lerp_(expected_logits, self.config.em_momentum)

        stats = {
            "nll": nll.detach(),
            "sparse_penalty": sparse_penalty.detach(),
            "mixture_weights": weights.detach(),
            "component_means": self.means.detach(),
            "component_variances": self.variances.detach(),
        }

        return loss, stats


__all__ = ["GMMConfig", "ResidualGMM"]
