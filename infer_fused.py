"""Prediction generator for the journal evaluation suite (Stage C, 2026-08).

Three inference variants per image, all on top of the FIXED infer_sam.CrackSAM:

  whole    - one pass on the image downscaled to <=1008 px long side; saves the
             raw instance set (scores + per-instance binary masks, packbits) as
             .npz so eval_metrics.py can re-threshold WITHOUT re-running the
             model (makes the threshold sweep nearly free)
  tilemax  - sliding-window tiles at native resolution, per-pixel logical OR
             (= max for binary masks) -> full-res PNG mask       [SS7.1 "max"]
  tilemean - sliding-window tiles, per-pixel vote fraction: crack if >=50% of
             the tiles covering the pixel said crack -> PNG mask [SS7.1 "mean"]

--base builds SAM3 with NO LoRA (the zero-shot benchmark row). CLAHE and the
model transform are identical across variants, so differences are attributable
to the fusion strategy alone.

Example (one fold, test split):
  python infer_fused.py --config configs/fold_RW20.yaml \
      --weights /workspace/outputs/lowo/fold_RW20/best_lora_weights.pt \
      --data_dir /workspace/folds/fold_RW20/test \
      --out_dir  /workspace/results/fold_RW20/preds_lora \
      --variants whole tilemax tilemean \
      --tile-size 1008 --tile-overlap 0.30 --det-threshold 0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

import yaml

from infer_sam import CrackSAM
from sam3.model_builder import build_sam3_image_model


def build_base_model(args):
    """CrackSAM-compatible object with NO LoRA (zero-shot benchmark).

    Mirrors CrackSAM.__init__ exactly, minus apply_lora/load_lora_weights, and
    reuses every CrackSAM method (predict, _preprocess, ...) unchanged."""
    m = CrackSAM.__new__(CrackSAM)
    with open(args.config) as f:
        m.config = yaml.safe_load(f)
    m.weights_path = None
    m.resolution = args.resolution
    m.detection_threshold = args.det_threshold
    m.nms_iou_threshold = args.nms_iou
    m.use_clahe = True
    m.clahe_clip_limit = 3.0
    m.clahe_tile_grid = 8
    m.use_postprocess = False
    m.pp_close_kernel = 20
    m.pp_open_kernel = 3
    m.use_skeleton = False
    m.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Building BASE SAM3 (zero-shot, no LoRA)...")
    m.model = build_sam3_image_model(
        device=m.device.type, compile=False, load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=True)
    m.model.to(m.device)
    m.model.eval()
    from sam3.train.transforms.basic_for_api import (  # same import as infer_sam
        ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI)
    m.transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=m.resolution, max_size=m.resolution, square=True,
                        consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    m.use_manual_postprocess = True
    return m


def downscale(pil, max_side):
    w, h = pil.size
    s = max_side / max(w, h)
    if s >= 1.0:
        return pil
    return pil.resize((int(round(w * s)), int(round(h * s))), PILImage.BILINEAR)


@torch.no_grad()
def whole_instances(model, pil, max_side):
    """One pass on the downscaled image -> (scores [K], masks [K,h,w] bool)."""
    small = downscale(pil, max_side)
    res = model.predict(small, text_prompts=["crack"], verbose=False)
    q = res[0]
    if q["num_detections"] == 0 or q["masks"] is None:
        h, w = small.size[1], small.size[0]
        return np.zeros(0, np.float32), np.zeros((0, h, w), bool)
    return q["scores"].astype(np.float32), q["masks"].astype(bool)


@torch.no_grad()
def tile_vote(model, pil, tile, overlap, mode):
    """Sliding window at native res; OR (tilemax) or >=50% vote (tilemean).

    CLAHE is applied ONCE on the full image (the SS8c-era fix in infer_sam);
    tiles crop from that canvas and predict() skips re-preprocessing."""
    W, H = pil.size
    stride = max(1, int(round(tile * (1.0 - overlap))))
    votes = np.zeros((H, W), np.uint16)
    cover = np.zeros((H, W), np.uint16)
    pre = model._preprocess(pil)
    xs = sorted(set(list(range(0, max(1, W - tile + 1), stride)) + [max(0, W - tile)]))
    ys = sorted(set(list(range(0, max(1, H - tile + 1), stride)) + [max(0, H - tile)]))
    for y0 in ys:
        for x0 in xs:
            crop = pre.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H)))
            res = model.predict(crop, text_prompts=["crack"],
                                apply_preprocess=False, apply_postprocess=False,
                                verbose=False)
            q = res[0]
            cw, ch = crop.size
            cover[y0:y0 + ch, x0:x0 + cw] += 1
            if q["num_detections"] and q["masks"] is not None:
                union = np.any(q["masks"], axis=0)
                votes[y0:y0 + ch, x0:x0 + cw] += union.astype(np.uint16)
    # one tiling pass serves both fusion rules
    out = {}
    if "tilemax" in mode:
        out["tilemax"] = votes > 0
    if "tilemean" in mode:
        out["tilemean"] = votes.astype(np.float32) / np.maximum(cover, 1) >= 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base", action="store_true", help="zero-shot SAM3 (no LoRA)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["whole"],
                    choices=["whole", "tilemax", "tilemean"])
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--tile-overlap", type=float, default=0.30)
    ap.add_argument("--det-threshold", type=float, default=0.05,
                    help="whole: keep a LOW floor so eval can sweep upward; "
                         "tile*: set to the threshold chosen on valid")
    ap.add_argument("--nms-iou", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    with open(data / "_annotations.coco.json") as f:
        coco = json.load(f)
    images = coco["images"][: args.limit or None]

    if args.base:
        model = build_base_model(args)
    else:
        model = CrackSAM(config_path=args.config, weights_path=args.weights,
                         resolution=args.resolution,
                         detection_threshold=args.det_threshold,
                         nms_iou_threshold=args.nms_iou,
                         use_clahe=True, use_postprocess=False,
                         use_skeleton=False, device=args.device)

    out = Path(args.out_dir)
    for v in args.variants:
        (out / v).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for im in tqdm(images, desc=f"infer {'+'.join(args.variants)}"):
        pil = PILImage.open(data / im["file_name"]).convert("RGB")
        stem = Path(im["file_name"]).stem
        if "whole" in args.variants:
            scores, masks = whole_instances(model, pil, args.resolution)
            np.savez_compressed(
                out / "whole" / f"{stem}.npz",
                scores=scores,
                masks=np.packbits(masks.reshape(-1)) if masks.size else
                np.zeros(0, np.uint8),
                mask_shape=np.array(masks.shape, np.int64),
                orig_size=np.array([im["height"], im["width"]], np.int64))
        tile_modes = [v for v in ("tilemax", "tilemean") if v in args.variants]
        if tile_modes:
            fused = tile_vote(model, pil, args.tile_size, args.tile_overlap,
                              tile_modes)
            for v, m in fused.items():
                PILImage.fromarray((m * 255).astype(np.uint8)).save(
                    out / v / f"{stem}.png", optimize=True)

    with open(out / "infer_meta.json", "w") as f:
        json.dump(dict(variants=args.variants, base=bool(args.base),
                       weights=args.weights, resolution=args.resolution,
                       tile=args.tile_size, overlap=args.tile_overlap,
                       det_threshold=args.det_threshold, nms_iou=args.nms_iou,
                       n_images=len(images)), f, indent=2)
    print("infer_fused done:", args.out_dir)


if __name__ == "__main__":
    main()
