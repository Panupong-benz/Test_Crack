# -*- coding: utf-8 -*-
"""Unified trainer for benchmark rows A2 (U-Net) / A3 (DeepLabv3+) / A4 (SegFormer).

Fairness (benchmark_protocol.md SS3): identical data pipeline (SegTileAdapter
-> production tiling/aug/weighted sampling), identical composite loss
(losses.py = SAM3 weights per-pixel), fixed seeds, same budget knobs.

Usage (one run = one row x fold x seed):
  python train_seg.py --arch unet --data <fold_dir> --out <run_dir> \
      --seed 0 --epochs 40 --batch 8 [--resume]

Checkpoint-resume is mandatory (interruptible instances): state saved every
--save-every steps to <out>/last.pt; --resume continues from it. Best model
(valid soft-clDice proxy) at <out>/best.pt. All config + timing to
<out>/run.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from seg_dataset import build_loaders
from losses import composite_loss

ARCHS = {
    "unet":        dict(cls="Unet",          encoder="resnet34"),
    "deeplabv3p":  dict(cls="DeepLabV3Plus", encoder="resnet50"),
    "segformer":   dict(cls="Segformer",     encoder="mit_b2"),
}


def build_model(arch: str):
    import segmentation_models_pytorch as smp
    spec = ARCHS[arch]
    cls = getattr(smp, spec["cls"])
    # in 0.5-normalized space; encoder weights imagenet (declared in protocol:
    # pretrained encoders allowed for every row incl. SAM3's pretrain)
    return cls(encoder_name=spec["encoder"], encoder_weights="imagenet",
               in_channels=3, classes=1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    """Soft clDice-style score on valid split (model-selection only —
    the reported metrics come from eval_masks.py, never from here)."""
    model.eval()
    inter = 0.0
    denom = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter += (p * y).sum().item() * 2
        denom += (p.sum() + y.sum()).item()
    model.train()
    return inter / max(denom, 1e-8)   # hard dice as cheap proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--data", required=True, help="fold dir with train/valid/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hard budget cap in optimizer steps (0 = epochs only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N steps then exit (smoke hour)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda"

    train, valid, _test, sampler = build_loaders(
        args.data, args.batch, args.tile_size, args.overlap,
        args.workers, args.seed)
    model = build_model(args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    step, epoch0, best = 0, 0, -1.0
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch0, best = ck["step"], ck["epoch"], ck["best"]
        print(f"[resume] step={step} epoch={epoch0} best={best:.4f}")

    cfg = vars(args) | {"arch_spec": ARCHS[args.arch],
                        "n_train_tiles": len(train.dataset),
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0)}
    (out / "run.json").write_text(json.dumps(cfg, indent=2))
    log = open(out / "train_log.csv", "a", encoding="utf-8")
    if log.tell() == 0:
        log.write("step,epoch,total,focal,dice,cldice,lr,sec\n")

    t0 = time.time()
    model.train()
    for epoch in range(epoch0, args.epochs):
        sampler.set_epoch(epoch)
        for batch in train:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = composite_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % 50 == 0:
                log.write(f"{step},{epoch},{parts['total']:.4f},"
                          f"{parts['focal']:.4f},{parts['dice']:.4f},"
                          f"{parts['cldice']:.4f},{args.lr},"
                          f"{time.time() - t0:.0f}\n")
                log.flush()
            if step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "step": step,
                            "epoch": epoch, "best": best}, last)
            if args.smoke and step >= args.smoke:
                print(f"[smoke] {args.smoke} steps in {time.time()-t0:.1f}s "
                      f"({(time.time()-t0)/args.smoke:.2f} s/step)")
                return
            if args.max_steps and step >= args.max_steps:
                break

        score = validate(model, valid, device)
        print(f"[epoch {epoch}] valid={score:.4f} best={best:.4f} step={step}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "step": step,
                        "epoch": epoch, "valid": score,
                        "arch": args.arch, "seed": args.seed}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "epoch": epoch + 1, "best": best}, last)
        if args.max_steps and step >= args.max_steps:
            break

    (out / "DONE").write_text(f"steps={step} best={best:.4f} "
                              f"hours={(time.time()-t0)/3600:.2f}")
    print(f"[done] best valid {best:.4f}, {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
