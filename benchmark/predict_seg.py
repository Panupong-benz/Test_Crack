# -*- coding: utf-8 -*-
"""Inference for benchmark rows A2/A3/A4: best.pt -> binary mask PNG per
test image, in the shape eval_masks.py consumes (<stem>_mask.png, 0/255).

Sliding window mirrors the production geometry (tile 1008, overlap 0.25 —
the same numbers infer_sam.py uses). Overlap fusion for these semantic rows
is MEAN of probabilities (declared in benchmark_protocol.md Amendment A1;
SS7.1's max-vs-mean study is a separate toggle and stays out of the
baseline rows). Threshold 0.5 on the fused probability.

Usage:
  python predict_seg.py --run <run_dir_with_best.pt> --fold <fold_dir> \
      --out <masks_dir> [--tile-size 1008] [--overlap 0.25] [--batch 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from train_seg import build_model

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def tile_starts(length: int, tile: int, stride: int):
    """Start offsets covering [0, length) with a final flush-to-edge tile."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_image(model, img_bgr: np.ndarray, device, tile: int,
                  stride: int, batch: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=0)
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5                      # production normalization
    prob = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    coords = [(y, x) for y in tile_starts(H, tile, stride)
              for x in tile_starts(W, tile, stride)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = np.stack([rgb[y:y + tile, x:x + tile] for y, x in chunk])
        t = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(t)
        if logits.shape[-2:] != (tile, tile):
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(tile, tile), mode="bilinear",
                align_corners=False)
        p = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for (y, x), pm in zip(chunk, p):
            prob[y:y + tile, x:x + tile] += pm
            count[y:y + tile, x:x + tile] += 1.0
    prob /= np.maximum(count, 1.0)               # MEAN fusion (declared)
    return (prob[:h, :w] > 0.5).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="run dir containing best.pt (from train_seg.py)")
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="predict only the first N images (0 = all). Used by "
                         "the smoke hour to exercise this path in minutes "
                         "instead of on 108 full-resolution frames.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.run / "best.pt", map_location=device)
    model = build_model(ck["arch"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    stride = max(1, int(round(args.tile_size * (1.0 - args.overlap))))
    img_dir = args.fold / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix in EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
        print(f"--limit {args.limit}: predicting {len(imgs)} of the split")
    args.out.mkdir(parents=True, exist_ok=True)
    total_sec, total_tiles = 0.0, 0
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{i+1}/{len(imgs)}] UNREADABLE {p.name}")
            continue
        h, w = img.shape[:2]
        n_tiles = (len(tile_starts(max(h, args.tile_size), args.tile_size, stride))
                   * len(tile_starts(max(w, args.tile_size), args.tile_size, stride)))
        t0 = time.time()
        mask = predict_image(model, img, device, args.tile_size, stride,
                             args.batch)
        total_sec += time.time() - t0
        total_tiles += n_tiles
        cv2.imwrite(str(args.out / f"{p.stem}_mask.png"), mask)
        print(f"[{i+1}/{len(imgs)}] {p.name} -> {p.stem}_mask.png "
              f"({(mask > 0).sum()} px)")
    (args.out / "predict_run.json").write_text(json.dumps({
        "run": str(args.run), "arch": ck.get("arch"),
        "seed": ck.get("seed"), "n_images": len(imgs),
        "tile_size": args.tile_size, "overlap": args.overlap,
        "fusion": "mean", "threshold": 0.5, "limit": args.limit,
        "ms_per_tile": round(1000.0 * total_sec / max(total_tiles, 1), 2),
        "n_tiles": total_tiles}, indent=2))


if __name__ == "__main__":
    main()
