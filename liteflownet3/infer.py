"""Inference script for LiteFlowNet3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from liteflownet3.models.liteflownet3 import LiteFlowNet3


def load_image(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        image = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0)


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

    image1 = load_image(args.image1).to(device)
    image2 = load_image(args.image2).to(device)

    with torch.no_grad():
        flow = model(image1, image2)

    save_flow(flow, args.output)
    print(f"Saved flow to {args.output}")


if __name__ == "__main__":
    main()

