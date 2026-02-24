# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script (SINGLE GPU)
- No torch.distributed
- No DDP
- Keeps original training logic: recon + LPIPS + GAN + EMA
- Works with your existing utils signatures:
    - prepare_dataloader(data_path, batch_size, num_workers, rank, world_size, transform)
    - build_scheduler(optimizer, steps_per_epoch, cfg_dict)
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from omegaconf import OmegaConf

from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs, prepare_dataloader
from utils.optim_utils import build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE (Single GPU).")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1/training/gan sections.")
    parser.add_argument("--data-path", type=str, required=True, help="Parquet glob or ImageFolder root (your prepare_dataloader handles it).")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def create_logger(logging_dir: str) -> logging.Logger:
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (rae_config, *_) = parse_configs(args.config)
    full_cfg = OmegaConf.load(args.config)

    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})

    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(training_cfg.get("num_workers", 8))
    clip_grad_val = training_cfg.get("clip_grad", 0.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 2500))
    ema_decay = float(training_cfg.get("ema_decay", 0.999))
    num_epochs = int(training_cfg.get("epochs", 8))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = os.path.join(args.results_dir, f"{experiment_index:03d}-RAE")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    if args.wandb:
        entity = os.environ.get("ENTITY", None)
        project = os.environ.get("PROJECT", None)
        if entity and project:
            wandb_utils.initialize(args, entity, f"{experiment_index:03d}-RAE", project)
        else:
            logger.info("W&B requested but ENTITY/PROJECT env not set. Skipping wandb init.")

    # -------------------------
    # model
    # -------------------------
    rae: RAE = instantiate_from_config(rae_config).to(device)

    use_vq = getattr(rae, "use_vq", False)
    if use_vq:
        logger.info("⚡ VQ Mode Enabled")

    # follow original behavior: freeze encoder, train decoder (+ VQ modules)
    rae.encoder.eval()
    rae.decoder.train()
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)

    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)

    # -------------------------
    # discriminator + lpips
    # -------------------------
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    lpips = LPIPS().to(device)
    lpips.eval()

    # -------------------------
    # data
    # -------------------------
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # IMPORTANT: call with POSITIONAL args (matches your util signature)
    loader, sampler = prepare_dataloader(
        args.data_path,
        batch_size,
        num_workers,
        0,   # rank
        1,   # world_size
        transform=transform,
    )

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    # -------------------------
    # optimizer / scheduler
    # -------------------------
    trainable_params = list(rae.decoder.parameters())
    if use_vq:
        trainable_params += list(rae.vq_pre.parameters())
        trainable_params += list(rae.vq_post.parameters())
        trainable_params += list(rae.vq_layer.parameters())

    optimizer, optim_msg = build_optimizer(trainable_params, training_cfg)

    scheduler: Optional[LambdaLR] = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)

    disc_scheduler: Optional[LambdaLR] = None
    disc_sched_msg: Optional[str] = None
    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

    logger.info(optim_msg)
    if sched_msg:
        logger.info(sched_msg)
    logger.info(disc_optim_msg)
    if disc_sched_msg:
        logger.info(disc_sched_msg)
    logger.info(f"Training for {num_epochs} epochs, batch size {batch_size}. steps/epoch={steps_per_epoch}")

    # -------------------------
    # precision
    # -------------------------
    scaler: Optional[GradScaler]
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)

    # -------------------------
    # resume (optional)
    # -------------------------
    start_epoch = 0
    global_step = 0
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                str(ckpt_path),
                rae,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step})")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -------------------------
    # training loop
    # -------------------------
    last_layer = rae.decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch

    for epoch in range(start_epoch, num_epochs):
        rae.train()
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0

        for step, (images, _) in enumerate(loader):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0

            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0

            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()

            with autocast(**autocast_kwargs):
                if use_vq:
                    recon, vq_loss = rae(images)   # training mode returns (recon, vq_loss)
                else:
                    recon = rae(images)            # training mode returns recon
                    vq_loss = torch.zeros((), device=device)

                recon_normed = recon * 2.0 - 1.0
                rec_loss = F.l1_loss(recon, images)

                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())

                recon_total = rec_loss + perceptual_weight * lpips_loss + vq_loss

                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)

            if use_gan:
                adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, max_d_weight)
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total

            if scaler is not None:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(rae.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(rae.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, rae, ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                rae.eval()
                discriminator.train()

                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)

                    with autocast(**autocast_kwargs):
                        with torch.no_grad():
                            if use_vq:
                                recon_disc = rae(images)  # eval mode returns recon only in your RAE
                            else:
                                recon_disc = rae(images)

                            recon_disc_normed = recon_disc * 2.0 - 1.0

                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0

                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)

                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)

                    if scaler is not None:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()

                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                    }

                    if disc_scheduler is not None:
                        disc_scheduler.step()

                discriminator.eval()
                rae.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["vq"] += vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else 0.0
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/vq": vq_loss.detach().item() if isinstance(vq_loss, torch.Tensor) else 0.0,
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "gan/weight": adaptive_weight.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)

            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                save_checkpoint(
                    ckpt_path,
                    global_step,
                    epoch,
                    rae,
                    ema_model,
                    optimizer,
                    scheduler,
                    discriminator,
                    disc_optimizer,
                    disc_scheduler,
                )

            global_step += 1

        if num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_vq = (epoch_metrics["vq"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(
                    [
                        f"loss_total: {avg_total:.4f}",
                        f"loss_recon: {avg_recon:.4f}",
                        f"loss_lpips: {avg_lpips:.4f}",
                        f"loss_vq: {avg_vq:.4f}",
                        f"loss_gan: {avg_gan:.4f}",
                    ]
                )
            )
            if args.wandb:
                wandb_utils.log(
                    {
                        "epoch/loss_total": avg_total,
                        "epoch/loss_recon": avg_recon,
                        "epoch/loss_lpips": avg_lpips,
                        "epoch/loss_vq": avg_vq,
                        "epoch/loss_gan": avg_gan,
                    },
                    step=global_step,
                )


if __name__ == "__main__":
    main()
