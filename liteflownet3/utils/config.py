"""Configuration helpers for LiteFlowNet3 training scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    mixed_precision: bool = True
    clip_grad_norm: float | None = 5.0


__all__ = ["OptimizerConfig", "TrainingConfig"]

