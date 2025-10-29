"""Training script for LiteFlowNet3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from liteflownet3.datasets.flying_chairs import FlowPairDataset, collate_flow_samples
from liteflownet3.models.gan import PatchDiscriminator
from liteflownet3.models.liteflownet3 import LiteFlowNet3
from liteflownet3.utils.config import OptimizerConfig, TrainingConfig
from liteflownet3.utils.train_utils import ensure_cuda_compat, save_checkpoint


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LiteFlowNet3 using PyTorch")
    parser.add_argument("list_file", help="Path to a training list file with image and flow triplets")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers, help="Data loader workers")
    parser.add_argument("--lr", type=float, default=OptimizerConfig.lr, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=OptimizerConfig.weight_decay, help="Optimizer weight decay")
    parser.add_argument("--disc-lr", type=float, default=OptimizerConfig.lr, help="Learning rate for the discriminator")
    parser.add_argument("--adv-weight", type=float, default=0.01, help="Weight for the adversarial loss term")
    parser.add_argument("--checkpoint-dir", type=str, default=TrainingConfig.checkpoint_dir, help="Directory for checkpoints")
    parser.add_argument("--log-interval", type=int, default=TrainingConfig.log_interval, help="Logging interval in iterations")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--clip-grad", type=float, default=TrainingConfig.clip_grad_norm, help="Gradient clipping norm (set <=0 to disable)")
    parser.add_argument(
        "--outlier-weight",
        type=float,
        default=0.1,
        help="Weight for the outlier-aware modulation loss",
    )
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
    discriminator = PatchDiscriminator().to(device)

    gen_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.disc_lr, weight_decay=args.weight_decay
    )

    gen_scaler = GradScaler(enabled=not args.no_amp)
    disc_scaler = GradScaler(enabled=not args.no_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in dataloader:
            image1 = batch.image1.to(device, non_blocking=True)
            image2 = batch.image2.to(device, non_blocking=True)
            flow_gt = batch.flow.to(device, non_blocking=True)

            gen_optimizer.zero_grad(set_to_none=True)

            discriminator.requires_grad_(False)
            with autocast(enabled=gen_scaler.is_enabled()):
                pred_flow = model(image1, image2)
                gmm_loss, loss_stats = model.residual_nll(pred_flow, flow_gt)
                residual_mag = torch.norm(pred_flow - flow_gt, dim=1, keepdim=True)
                gating_maps = model.latest_gating()

                outlier_map = gating_maps["outlier"]
                if outlier_map.shape[-2:] != residual_mag.shape[-2:]:
                    outlier_map = F.interpolate(
                        outlier_map,
                        size=residual_mag.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )

                confidence_map = gating_maps["confidence"]
                if confidence_map.shape[-2:] != residual_mag.shape[-2:]:
                    confidence_map = F.interpolate(
                        confidence_map,
                        size=residual_mag.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )

                scale = residual_mag.detach().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                outlier_targets = 1.0 - torch.exp(-residual_mag / scale)
                outlier_loss = F.binary_cross_entropy(outlier_map, outlier_targets)

                fake_logits = discriminator(residual_mag)
                adv_targets = torch.ones_like(fake_logits)
                adv_loss = F.binary_cross_entropy_with_logits(fake_logits, adv_targets)
                total_loss = gmm_loss + args.adv_weight * adv_loss + args.outlier_weight * outlier_loss

            gen_scaler.scale(total_loss).backward()

            if args.clip_grad and args.clip_grad > 0:
                gen_scaler.unscale_(gen_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            gen_scaler.step(gen_optimizer)
            gen_scaler.update()

            discriminator.requires_grad_(True)
            disc_optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=disc_scaler.is_enabled()):
                residual_detached = residual_mag.detach()
                fake_logits_detached = discriminator(residual_detached)
                real_residual = torch.zeros_like(residual_detached)
                real_logits = discriminator(real_residual)
                real_targets = torch.ones_like(real_logits)
                fake_targets = torch.zeros_like(fake_logits_detached)
                d_loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)
                d_loss_fake = F.binary_cross_entropy_with_logits(fake_logits_detached, fake_targets)
                disc_loss = 0.5 * (d_loss_real + d_loss_fake)

            disc_scaler.scale(disc_loss).backward()
            disc_scaler.step(disc_optimizer)
            disc_scaler.update()

            loss_stats = {
                **loss_stats,
                "adv_loss": adv_loss.detach(),
                "disc_loss": disc_loss.detach(),
                "total_loss": total_loss.detach(),
                "outlier_loss": outlier_loss.detach(),
                "confidence_mean": confidence_map.mean().detach(),
                "outlier_mean": outlier_map.mean().detach(),
            }

            if global_step % args.log_interval == 0:
                weights = loss_stats["mixture_weights"].detach().cpu().tolist()
                mixture = ", ".join(f"{w:.3f}" for w in weights)
                print(
                    f"Epoch {epoch} Iter {global_step}: "
                    f"total={loss_stats['total_loss'].item():.4f} "
                    f"(nll={loss_stats['nll'].item():.4f}, adv={loss_stats['adv_loss'].item():.4f}, "
                    f"disc={loss_stats['disc_loss'].item():.4f}, outlier={loss_stats['outlier_loss'].item():.4f}, "
                    f"sparse={loss_stats['sparse_penalty'].item():.4f}, conf={loss_stats['confidence_mean'].item():.3f}, "
                    f"anom={loss_stats['outlier_mean'].item():.3f}, "
                    f"mix=[{mixture}])"
                )

            global_step += 1

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            gen_optimizer,
            epoch,
            extra={
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
            },
        )
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

