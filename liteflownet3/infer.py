"""Inference script for LiteFlowNet3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from liteflownet3.models.liteflownet3 import LiteFlowNet3


def load_image(path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(path) as img:
        image = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    tensor = image.unsqueeze(0)
    _, h, w = tensor.shape
    return tensor, (h, w)


def pad_to_multiple(
    image: torch.Tensor, multiple: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad a BCHW tensor so H and W become multiples of ``multiple``."""

    _, _, h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0, 0)

    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    padded = F.pad(image, padding, mode="replicate")
    return padded, padding


def save_flow(flow: torch.Tensor, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np.save(path_obj, flow_np)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LiteFlowNet3 inference on two images")
    parser.add_argument("image1", help="Path to the first RGB image")
    parser.add_argument("image2", help="Path to the second RGB image")
    parser.add_argument("checkpoint", help="Path to a trained model checkpoint (.pt)")
    parser.add_argument("--output", default="flow.npy", help="Where to save the resulting flow (NumPy format)")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda or cpu)")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = LiteFlowNet3()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    image1, size1 = load_image(args.image1)
    image2, size2 = load_image(args.image2)

    if size1 != size2:
        raise ValueError(
            f"Input images must share the same resolution, but got {size1} and {size2}."
        )

    image1 = image1.to(device)
    image2 = image2.to(device)

    image1, padding = pad_to_multiple(image1)
    image2, _ = pad_to_multiple(image2)

    with torch.no_grad():
        flow = model(image1, image2)

    if padding != (0, 0, 0, 0):
        left, right, top, bottom = padding
        _, _, padded_h, padded_w = flow.shape
        flow = flow[..., top : padded_h - bottom, left : padded_w - right]
        orig_h, orig_w = size1
        flow = flow[..., :orig_h, :orig_w]

    save_flow(flow, args.output)
    print(f"Saved flow to {args.output}")


if __name__ == "__main__":
    main()

