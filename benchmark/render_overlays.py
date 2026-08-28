# -*- coding: utf-8 -*-
"""FP/FN overlay renderer + qualitative model-comparison panels
(benchmark_protocol.md Amendment A1.2: overlay = TP green / FP red /
FN blue on the photo; panels = same crop across all model rows).

Two jobs, one color code:
  1) Error analysis (model improvement): --worst N reads
     per_image_metrics.csv, picks each model's N lowest-scoring test
     images, renders their overlays -> the most direct starting point
     for "why did this model fail here".
  2) Paper qualitative figure: --panel renders one row per image x one
     column per model (crop via --crop x,y,w,h), for the 4 declared crop
     types (clear crack / grid line / written number / dense zone).

GT comes from eval_masks.load_gt_masks (single GT definition — never
re-decoded here).

Usage:
  python render_overlays.py overlay --fold <fold> --pred runs/<tag>/masks \
      --out overlays/<tag> [--images IMG_A.jpg IMG_B.jpg]
  python render_overlays.py worst --fold <fold> --per-image \
      results/benchmark/per_image_metrics.csv --model a6 --seed 0 -n 5 \
      --runs-dir runs --out overlays/worst_a6
  python render_overlays.py panel --fold <fold> --image IMG_4128.jpg \
      --runs-dir runs --models a1 unet deeplabv3p segformer a5 a6 \
      --seed 0 --crop 800,1200,1000,1000 --out panels/grid_line.png
  python render_overlays.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_masks import load_gt_masks  # noqa: E402

# BGR
C_TP, C_FP, C_FN = (80, 200, 80), (60, 60, 230), (230, 120, 60)
ALPHA = 0.65
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_pred(pred_dir: Path, name: str):
    stem = Path(name).stem
    for c in (pred_dir / name, pred_dir / f"{stem}.png",
              pred_dir / f"{stem}_mask.png"):
        if c.exists():
            return c
    return None


def overlay_image(img: np.ndarray, pred: np.ndarray,
                  gt: np.ndarray) -> np.ndarray:
    """Paint TP/FP/FN classes over the photo. Pure function (selftested)."""
    p = pred > 0
    g = gt > 0
    out = img.copy()
    for m, color in (((p & g), C_TP), ((p & ~g), C_FP), ((~p & g), C_FN)):
        out[m] = (ALPHA * np.array(color)
                  + (1 - ALPHA) * out[m]).astype(np.uint8)
    return out


def load_pair(fold: Path, pred_dir: Path, name: str, gts: dict):
    img_p = fold / "test" / name
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None
    gt = gts[name]
    pf = find_pred(pred_dir, name)
    pred = (cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
            if pf is not None else np.zeros_like(gt))
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return img, pred, gt


def cmd_overlay(args):
    gts = load_gt_masks(args.fold / "test")
    names = args.images or sorted(gts)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        img, pred, gt = load_pair(args.fold, args.pred, name, gts)
        if img is None:
            print(f"SKIP unreadable {name}")
            continue
        cv2.imwrite(str(args.out / f"{Path(name).stem}_overlay.png"),
                    overlay_image(img, pred, gt))
        print(f"overlay {name}")


def cmd_worst(args):
    rows = [r for r in csv.DictReader(open(args.per_image, encoding="utf-8"))
            if r["model"] == args.model and str(r["seed"]) == str(args.seed)
            and r["fold"] == args.fold.name.replace("fold_", "")]
    rows.sort(key=lambda r: float(r[args.metric]))
    picks = rows[:args.n]
    pred_dir = args.runs_dir / (
        f"{args.model}_{args.fold.name.replace('fold_', '')}"
        + (f"_s{args.seed}" if args.seed != "" else "")) / "masks"
    gts = load_gt_masks(args.fold / "test")
    args.out.mkdir(parents=True, exist_ok=True)
    for r in picks:
        img, pred, gt = load_pair(args.fold, pred_dir, r["image"], gts)
        if img is None:
            continue
        out = overlay_image(img, pred, gt)
        cv2.putText(out, f"{args.metric}={float(r[args.metric]):.3f}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
        cv2.imwrite(str(args.out
                        / f"{Path(r['image']).stem}_overlay.png"), out)
        print(f"worst: {r['image']}  {args.metric}={r[args.metric]}")


def cmd_panel(args):
    gts = load_gt_masks(args.fold / "test")
    fold_wall = args.fold.name.replace("fold_", "")
    tiles = []
    x, y, w, h = (map(int, args.crop.split(",")) if args.crop
                  else (0, 0, 0, 0))
    for model in args.models:
        seedless = model in ("a1", "a5")
        tag = f"{model}_{fold_wall}" + ("" if seedless
                                        else f"_s{args.seed}")
        img, pred, gt = load_pair(args.fold, args.runs_dir / tag / "masks",
                                  args.image, gts)
        if img is None:
            print(f"SKIP {model}: image unreadable")
            continue
        ov = overlay_image(img, pred, gt)
        if args.crop:
            ov = ov[y:y + h, x:x + w]
        cv2.putText(ov, model, (12, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 0), 3)
        tiles.append(ov)
    if not tiles:
        print("no tiles rendered")
        return 1
    hh = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * hh / t.shape[0]), hh))
             for t in tiles]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.hstack(tiles))
    print(f"panel -> {args.out} ({len(tiles)} models)")
    return 0


def selftest():
    """Planted 3-class masks -> exact painted-pixel counts per class."""
    img = np.full((64, 64, 3), 255, np.uint8)
    gt = np.zeros((64, 64), np.uint8)
    pred = np.zeros((64, 64), np.uint8)
    gt[10:20, 10:20] = 1                       # 100 GT px
    pred[10:20, 15:25] = 1                     # 100 pred px, 50 overlap
    out = overlay_image(img, pred, gt)
    tp_px = int((out == np.array(
        (ALPHA * np.array(C_TP) + (1 - ALPHA) * 255).astype(np.uint8)
    )).all(axis=2).sum())
    fp_px = int((out == np.array(
        (ALPHA * np.array(C_FP) + (1 - ALPHA) * 255).astype(np.uint8)
    )).all(axis=2).sum())
    fn_px = int((out == np.array(
        (ALPHA * np.array(C_FN) + (1 - ALPHA) * 255).astype(np.uint8)
    )).all(axis=2).sum())
    assert (tp_px, fp_px, fn_px) == (50, 50, 50), (tp_px, fp_px, fn_px)
    untouched = int((out == 255).all(axis=2).sum())
    assert untouched == 64 * 64 - 150, untouched
    print("selftest PASS: TP/FP/FN painted-pixel counts exact (50/50/50), "
          "background untouched")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    o = sub.add_parser("overlay")
    o.add_argument("--fold", type=Path, required=True)
    o.add_argument("--pred", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--images", nargs="+", default=None)

    w = sub.add_parser("worst")
    w.add_argument("--fold", type=Path, required=True)
    w.add_argument("--per-image", type=Path, required=True)
    w.add_argument("--model", required=True)
    w.add_argument("--seed", default="0")
    w.add_argument("-n", type=int, default=5)
    w.add_argument("--metric", default="cliou_4px")
    w.add_argument("--runs-dir", type=Path, default=Path("runs"))
    w.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("panel")
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seed", default="0")
    p.add_argument("--crop", default=None, help="x,y,w,h in image px")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "overlay":
        return cmd_overlay(args)
    if args.cmd == "worst":
        return cmd_worst(args)
    if args.cmd == "panel":
        return cmd_panel(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
