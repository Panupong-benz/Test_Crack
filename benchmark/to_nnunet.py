# -*- coding: utf-8 -*-
"""Convert one LOWO fold into nnU-Net v2 raw format (row A1).

Deterministic tiles (no jitter, no augment — nnU-Net does its own aug; giving
it pre-augmented data would double-augment) at the same 1008/0.25 grid as
every other row. Train+valid of the fold -> imagesTr/labelsTr (nnU-Net makes
its own internal split; we train with `--split all` wait-free since model
selection across our valid split is not what nnU-Net does natively — declared
in protocol SS3 as nnU-Net's self-configuring identity). Test images are NOT
tiled: full-frame inference at eval time via sliding window is nnU-Net's own
mechanism, so imagesTs receives the full test frames.

Budget parity: train with nnUNetTrainer_250epochs (built-in variant), the cap
declared in benchmark_protocol Amendment A1 before any run.

Usage:
  python to_nnunet.py --fold <fold_dir> --dataset-id 501 \
      --name BMfoldRW20 --raw <nnUNet_raw_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_masks import load_gt_masks  # noqa: E402  (same GT decoding everywhere)

TILE, OVERLAP = 1008, 0.25


def tile_origins(w, h):
    stride = max(1, int(round(TILE * (1 - OVERLAP))))
    xs = list(range(0, max(1, w - TILE + 1), stride))
    ys = list(range(0, max(1, h - TILE + 1), stride))
    if w > TILE and xs[-1] != w - TILE:
        xs.append(w - TILE)
    if h > TILE and ys[-1] != h - TILE:
        ys.append(h - TILE)
    return [(x, y) for y in ys for x in xs] or [(0, 0)]


def crop_pad(arr, x0, y0, fill=0):
    h, w = arr.shape[:2]
    out_shape = (TILE, TILE) + arr.shape[2:]
    out = np.full(out_shape, fill, dtype=arr.dtype)
    x1, y1 = min(x0 + TILE, w), min(y0 + TILE, h)
    out[: y1 - y0, : x1 - x0] = arr[y0:y1, x0:x1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", type=Path, required=True,
                    help="nnUNet_raw root")
    args = ap.parse_args()

    ds = args.raw / f"Dataset{args.dataset_id:03d}_{args.name}"
    (ds / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds / "labelsTr").mkdir(exist_ok=True)
    (ds / "imagesTs").mkdir(exist_ok=True)

    n_tr = 0
    for split in ("train", "valid"):
        gts = load_gt_masks(args.fold / split)
        for name, gt in sorted(gts.items()):
            img = cv2.imread(str(args.fold / split / name))
            if img is None:
                print(f"WARNING missing image {split}/{name}")
                continue
            fill = int(img.mean())
            for i, (x0, y0) in enumerate(tile_origins(*img.shape[1::-1])):
                stem = f"{Path(name).stem}_t{i:02d}"
                cv2.imwrite(str(ds / "imagesTr" / f"{stem}_0000.png"),
                            crop_pad(img, x0, y0, fill))
                cv2.imwrite(str(ds / "labelsTr" / f"{stem}.png"),
                            crop_pad(gt, x0, y0, 0))
                n_tr += 1

    gts_ts = load_gt_masks(args.fold / "test")
    for name in sorted(gts_ts):
        img = cv2.imread(str(args.fold / "test" / name))
        cv2.imwrite(str(ds / "imagesTs" / f"{Path(name).stem}_0000.png"), img)

    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "RGB"},
        "labels": {"background": 0, "crack": 1},
        "numTraining": n_tr,
        "file_ending": ".png",
    }, indent=2))
    print(f"{ds}: {n_tr} training tiles, {len(gts_ts)} test frames")


if __name__ == "__main__":
    main()
