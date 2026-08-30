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
  python to_nnunet.py --selftest
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


def convert(fold: Path, dataset_id: int, name: str, raw: Path) -> Path:
    ds = raw / f"Dataset{dataset_id:03d}_{name}"
    (ds / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds / "labelsTr").mkdir(exist_ok=True)
    (ds / "imagesTs").mkdir(exist_ok=True)

    n_tr = 0
    for split in ("train", "valid"):
        gts = load_gt_masks(fold / split)
        for nm, gt in sorted(gts.items()):
            img = cv2.imread(str(fold / split / nm))
            if img is None:
                print(f"WARNING missing image {split}/{nm}")
                continue
            fill = int(img.mean())
            for i, (x0, y0) in enumerate(tile_origins(*img.shape[1::-1])):
                stem = f"{Path(nm).stem}_t{i:02d}"
                cv2.imwrite(str(ds / "imagesTr" / f"{stem}_0000.png"),
                            crop_pad(img, x0, y0, fill))
                # load_gt_masks returns 0/1, which is what "labels" below
                # declares; writing 0/255 here would fail nnU-Net's
                # --verify_dataset_integrity. Pinned by selftest().
                cv2.imwrite(str(ds / "labelsTr" / f"{stem}.png"),
                            crop_pad(gt, x0, y0, 0))
                n_tr += 1

    gts_ts = load_gt_masks(fold / "test")
    for nm in sorted(gts_ts):
        img = cv2.imread(str(fold / "test" / nm))
        cv2.imwrite(str(ds / "imagesTs" / f"{Path(nm).stem}_0000.png"), img)

    (ds / "dataset.json").write_text(json.dumps({
        # 3 entries, not 1: cv2 writes a 3-channel PNG and nnU-Net's
        # NaturalImage2DIO returns 3 channels, so a single "RGB" key
        # fails --verify_dataset_integrity (Amendment A1.4)
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "crack": 1},
        "numTraining": n_tr,
        "file_ending": ".png",
    }, indent=2))
    print(f"{ds}: {n_tr} training tiles, {len(gts_ts)} test frames")
    return ds


def _fake_split(d: Path, names, with_poly=True):
    d.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for i, nm in enumerate(names):
        cv2.imwrite(str(d / nm), np.full((150, 200, 3), 90, np.uint8))
        images.append({"id": i, "file_name": nm, "width": 200, "height": 150})
        if with_poly:
            anns.append({"id": i, "image_id": i, "category_id": 1,
                         "segmentation": [[10, 10, 60, 10, 60, 20, 10, 20]],
                         "bbox": [10, 10, 50, 10], "area": 500,
                         "iscrowd": 0})
    (d / "_annotations.coco.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 1, "name": "crack"}]}), encoding="utf-8")


def selftest():
    """to_nnunet's only other execution is the LAST block of the paid queue,
    so a bad dataset.json or label range would surface after every training
    hour was already spent (Amendment A1.5). This builds a two-image fold,
    converts it, and pins what nnU-Net actually validates."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fold = root / "fold_X"
        _fake_split(fold / "train", ["a.png", "b.png"])
        _fake_split(fold / "valid", ["c.png"])
        _fake_split(fold / "test", ["d.png"])
        ds = convert(fold, 599, "SELFTEST", root / "raw")

        imgs = sorted((ds / "imagesTr").glob("*.png"))
        labs = sorted((ds / "labelsTr").glob("*.png"))
        ts = sorted((ds / "imagesTs").glob("*.png"))
        meta = json.loads((ds / "dataset.json").read_text())

        assert len(imgs) == len(labs) == 3, (len(imgs), len(labs))
        assert len(ts) == 1, len(ts)
        assert all(f.name.endswith("_0000.png") for f in imgs), (
            "nnU-Net requires the _0000 channel suffix on imagesTr")
        assert all(not f.name.endswith("_0000.png") for f in labs), (
            "labelsTr must NOT carry the channel suffix")
        assert ts[0].name.endswith("_0000.png"), ts[0].name

        im = cv2.imread(str(imgs[0]))
        assert im.shape == (TILE, TILE, 3), im.shape
        assert len(meta["channel_names"]) == im.shape[2] == 3, meta
        assert meta["numTraining"] == len(imgs), meta
        assert meta["file_ending"] == ".png"

        lab = cv2.imread(str(labs[0]), cv2.IMREAD_UNCHANGED)
        assert lab.shape == (TILE, TILE), lab.shape
        vals = set(np.unique(lab).tolist())
        assert vals <= set(meta["labels"].values()), (
            f"label values {sorted(vals)} are outside the declared "
            f"{meta['labels']} - nnU-Net --verify_dataset_integrity "
            f"would reject this dataset")
        assert 1 in vals, "the planted polygon produced no foreground"
    print("selftest PASS: 3 tiles + 1 test frame, _0000 suffix on images "
          "only, labels are 0/1 as dataset.json declares, 3 channel_names")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=Path)
    ap.add_argument("--dataset-id", type=int)
    ap.add_argument("--name")
    ap.add_argument("--raw", type=Path, help="nnUNet_raw root")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    missing = [k for k in ("fold", "dataset_id", "name", "raw")
               if getattr(args, k) is None]
    if missing:
        ap.error("required unless --selftest: " + ", ".join(missing))
    convert(args.fold, args.dataset_id, args.name, args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
