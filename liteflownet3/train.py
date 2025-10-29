"""Training script for LiteFlowNet3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from liteflownet3.datasets.flying_chairs import FlowPairDataset, collate_flow_samples
from liteflownet3.models.liteflownet3 import LiteFlowNet3
from liteflownet3.utils.config import OptimizerConfig, TrainingConfig
from liteflownet3.utils.train_utils import EndpointError, ensure_cuda_compat, save_checkpoint


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LiteFlowNet3 using PyTorch")
    parser.add_argument("list_file", help="Path to a training list file with image and flow triplets")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers, help="Data loader workers")
    parser.add_argument("--lr", type=float, default=OptimizerConfig.lr, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=OptimizerConfig.weight_decay, help="Optimizer weight decay")
    parser.add_argument("--checkpoint-dir", type=str, default=TrainingConfig.checkpoint_dir, help="Directory for checkpoints")
    parser.add_argument("--log-interval", type=int, default=TrainingConfig.log_interval, help="Logging interval in iterations")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--clip-grad", type=float, default=TrainingConfig.clip_grad_norm, help="Gradient clipping norm (set <=0 to disable)")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    ensure_cuda_compat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FlowPairDataset(args.list_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_flow_samples,
    )

    model = LiteFlowNet3().to(device)
    criterion = EndpointError().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=not args.no_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in dataloader:
            image1 = batch.image1.to(device, non_blocking=True)
            image2 = batch.image2.to(device, non_blocking=True)
            flow_gt = batch.flow.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                pred_flow = model(image1, image2)
                loss = criterion(pred_flow, flow_gt)

            scaler.scale(loss).backward()

            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} Iter {global_step}: loss={loss.item():.4f}")

            global_step += 1

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, epoch)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

