"""Datasets utilities for optical flow training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _read_flo(path: str) -> np.ndarray:
    """Read a .flo optical flow file into a numpy array."""

    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        if tag != 202021.25:
            raise ValueError(f"Invalid .flo file {path}: wrong tag {tag}")
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
    return data.reshape(height, width, 2)


def _pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


@dataclass
class FlowSample:
    image1: Tensor
    image2: Tensor
    flow: Tensor


class FlowPairDataset(Dataset[FlowSample]):
    """Dataset that returns pairs of images and their optical flow."""

    def __init__(
        self,
        list_file: str,
        transform: Optional[Callable[[FlowSample], FlowSample]] = None,
        loader: Callable[[str], Image.Image] = _pil_loader,
    ) -> None:
        super().__init__()
        self.items = self._parse_list(list_file)
        self.transform = transform
        self.loader = loader

    @staticmethod
    def _parse_list(list_file: str) -> List[Tuple[str, str, str]]:
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file {list_file} not found")

        directory = os.path.dirname(os.path.abspath(list_file))
        items: List[Tuple[str, str, str]] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid line in {list_file}: {line}")
                img1, img2, flow = parts
                items.append((os.path.join(directory, img1), os.path.join(directory, img2), os.path.join(directory, flow)))
        if not items:
            raise ValueError(f"No entries found in {list_file}")
        return items

    def __len__(self) -> int:  # noqa: D401 - refer to base class.
        return len(self.items)

    def __getitem__(self, idx: int) -> FlowSample:  # noqa: D401 - refer to base class.
        img1_path, img2_path, flow_path = self.items[idx]
        image1 = torch.from_numpy(np.array(self.loader(img1_path))).permute(2, 0, 1).float() / 255.0
        image2 = torch.from_numpy(np.array(self.loader(img2_path))).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(_read_flo(flow_path)).permute(2, 0, 1).float()

        sample = FlowSample(image1=image1, image2=image2, flow=flow)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_flow_samples(batch: List[FlowSample]) -> FlowSample:
    image1 = torch.stack([item.image1 for item in batch], dim=0)
    image2 = torch.stack([item.image2 for item in batch], dim=0)
    flow = torch.stack([item.flow for item in batch], dim=0)
    return FlowSample(image1=image1, image2=image2, flow=flow)


__all__ = ["FlowPairDataset", "FlowSample", "collate_flow_samples"]

