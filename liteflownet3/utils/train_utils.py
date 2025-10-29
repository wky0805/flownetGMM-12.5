"""Training utilities for LiteFlowNet3."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


class EndpointError(nn.Module):
    """End-point error between predicted and target flow fields."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:  # noqa: D401 - refer to base class.
        return torch.norm(pred - target, dim=1, keepdim=True).mean()


def ensure_cuda_compat(min_version: str = "12.4") -> None:
    """Ensure that the available CUDA runtime satisfies the minimum version."""

    if not torch.cuda.is_available():
        return

    version = torch.version.cuda
    if version is None:
        raise RuntimeError("CUDA is reported as available but torch.version.cuda is None")

    def _parse(v: str) -> Iterable[int]:
        return (int(part) for part in v.split("."))

    current = tuple(_parse(version))
    required = tuple(_parse(min_version))

    if current < required:
        raise RuntimeError(
            f"CUDA {version} detected, but LiteFlowNet3 requires at least CUDA {min_version}."
        )


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


__all__ = ["EndpointError", "ensure_cuda_compat", "save_checkpoint"]

