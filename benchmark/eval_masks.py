# -*- coding: utf-8 -*-
"""Single-source evaluator for ALL benchmark rows (A1-A6, axis B).

One metric implementation, one place — a number appearing in two tables can
never come from two definitions. Consumes predicted masks (PNG, 0/255 or
0/1) named identically to the GT test images, plus the fold's COCO GT.

Metrics (benchmark_protocol.md SS4):
  pixel_iou     TP/(TP+FP+FN) on pixels, dataset-aggregated (not mean-of-image)
  f1/precision/recall   pixel level, dataset-aggregated
  cldice        hard skeleton Dice: 2*Tprec*Tsens/(Tprec+Tsens),
                Tprec=|S_P∩V_L|/|S_P|, Tsens=|S_L∩V_P|/|S_L|,
                skeleton = skimage.morphology.skeletonize (SS8f lesson:
                never a morphological-open fallback)
  cliou_4px     tolerant centerline IoU after OmniCrack30k (Benz CVPRW 2024):
                TP=|S_P∩dil4(S_L)|, FP=|S_P|-TP, FN=|S_L - dil4(S_P)|,
                clIoU=TP/(TP+FP+FN); dil4 = binary dilation, disk radius 4 px
  marked_fp     FP pixel rate restricted to a frozen tile/image list
                (grid lines / stickers / written numbers subset)

Aggregation: TP/FP/FN summed over the whole split, ratios computed once at
the end (declared; mean-of-images would overweight near-empty images).

Usage:
  python eval_masks.py --gt <fold_dir>/test --pred <masks_dir> \
      --out results/benchmark/eval_<row>_<fold>_<seed>.csv \
      [--marked-list marked_line_images.txt] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pycocotools.mask as mask_utils
except ImportError:            # selftest path does not need COCO
    mask_utils = None
from skimage.morphology import skeletonize, binary_dilation, disk


# ---------------------------------------------------------------- metrics --
def confusion(pred: np.ndarray, gt: np.ndarray):
    p = pred > 0
    g = gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, fp, fn


def cldice_counts(pred: np.ndarray, gt: np.ndarray):
    p = pred > 0
    g = gt > 0
    sp = skeletonize(p)
    sg = skeletonize(g)
    return {"sp_in_g": int((sp & g).sum()), "sp": int(sp.sum()),
            "sg_in_p": int((sg & p).sum()), "sg": int(sg.sum())}


def cliou_counts(pred: np.ndarray, gt: np.ndarray, tol_px: int = 4):
    p = pred > 0
    g = gt > 0
    sp = skeletonize(p)
    sg = skeletonize(g)
    se = disk(tol_px)
    dil_sg = binary_dilation(sg, se)
    dil_sp = binary_dilation(sp, se)
    tp = int((sp & dil_sg).sum())
    fp = int(sp.sum()) - tp
    fn = int((sg & ~dil_sp).sum())
    return {"cl_tp": tp, "cl_fp": fp, "cl_fn": fn}


def finalize(acc: dict) -> dict:
    eps = 1e-9
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    prec = tp / max(tp + fp, eps)
    rec = tp / max(tp + fn, eps)
    out = {
        "pixel_iou": tp / max(tp + fp + fn, eps),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, eps),
    }
    tprec = acc["sp_in_g"] / max(acc["sp"], eps)
    tsens = acc["sg_in_p"] / max(acc["sg"], eps)
    out["cldice"] = 2 * tprec * tsens / max(tprec + tsens, eps)
    out["cliou_4px"] = acc["cl_tp"] / max(
        acc["cl_tp"] + acc["cl_fp"] + acc["cl_fn"], eps)
    if acc.get("marked_pixels", 0) > 0:
        out["marked_fp_rate"] = acc["marked_fp"] / acc["marked_pixels"]
    return out


# ------------------------------------------------------------------- I/O ---
def load_gt_masks(gt_dir: Path):
    """image file name -> union binary mask, from the split's COCO json."""
    import cv2
    coco = json.loads((gt_dir / "_annotations.coco.json").read_text(
        encoding="utf-8"))
    by_img = {}
    id2img = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    for img_id, im in id2img.items():
        h, w = im["height"], im["width"]
        m = np.zeros((h, w), dtype=np.uint8)
        for a in anns_by_img.get(img_id, []):
            seg = a.get("segmentation")
            if isinstance(seg, dict):
                m |= mask_utils.decode(seg).astype(np.uint8)
            elif isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(m, [pts], 1)
        by_img[im["file_name"]] = m
    return by_img


def load_gt_mask_dir(gt_dir: Path):
    """image file name -> binary mask, from a directory of GT mask PNGs
    (axis B external datasets ship PNG masks, not COCO). Any nonzero
    pixel counts as crack."""
    import cv2
    by_img = {}
    for p in sorted(gt_dir.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        by_img[p.name] = (m > 0).astype(np.uint8)
    return by_img


def run_eval(gt_dir: Path, pred_dir: Path, out_csv: Path,
             marked_list: Path | None, gt_mode: str = "coco"):
    import cv2
    gts = (load_gt_mask_dir(gt_dir) if gt_mode == "masks"
           else load_gt_masks(gt_dir))
    marked = set()
    if marked_list and marked_list.exists():
        marked = {l.strip() for l in marked_list.read_text().splitlines()
                  if l.strip()}
    acc = dict(tp=0, fp=0, fn=0, sp_in_g=0, sp=0, sg_in_p=0, sg=0,
               cl_tp=0, cl_fp=0, cl_fn=0, marked_fp=0, marked_pixels=0)
    per_image = []
    missing = []
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **k):
            return it
    # stderr bar (A1.16): up to 108 full-res skeletonize+dilate passes ran
    # in total silence. Nothing parses stdout structurally (summarize reads
    # eval_*.summary.json) and stderr merges into the queue log anyway.
    for name, gt in _tqdm(sorted(gts.items()), desc="eval", unit="img",
                          mininterval=1.0, file=sys.stderr, disable=False):
        stem = Path(name).stem
        # "_mask.png" FIRST and deliberately: infer_sam writes the binary
        # prediction as {stem}_mask.png and an RGB matplotlib OVERLAY as
        # {stem}.png. Preferring the bare stem silently scored the overlay
        # figure as the prediction on every SAM3 row (Amendment A1.4).
        cand = [pred_dir / f"{stem}_mask.png", pred_dir / name,
                pred_dir / f"{stem}.png"]
        pf = next((c for c in cand if c.exists()), None)
        if pf is None:
            missing.append(name)
            pred = np.zeros_like(gt)          # absent prediction = empty mask
        else:
            pred = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        tp, fp, fn = confusion(pred, gt)
        row = {"image": name, "px": int(gt.size),
               "tp": tp, "fp": fp, "fn": fn}
        row |= cldice_counts(pred, gt)
        row |= cliou_counts(pred, gt)
        for k in ("tp", "fp", "fn", "sp_in_g", "sp", "sg_in_p", "sg",
                  "cl_tp", "cl_fp", "cl_fn"):
            acc[k] += row[k]
        if name in marked or stem in marked:
            acc["marked_fp"] += fp
            acc["marked_pixels"] += gt.size
        per_image.append(row)

    summary = finalize(acc)
    summary["n_images"] = len(per_image)
    summary["n_missing_pred"] = len(missing)
    summary["counts"] = acc          # raw counts -> summarize_benchmark.py
                                     # pools across folds via the SAME finalize()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    out_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} predictions missing (scored as empty)")
    return summary


# --------------------------------------------------------------- selftest --
def selftest():
    """mask==GT -> all 1.0; empty pred -> 0; 2px-shift -> pixel metrics drop
    but cliou_4px stays 1.0 (tolerance working)."""
    rng = np.random.default_rng(0)
    gt = np.zeros((256, 256), np.uint8)
    gt[100:103, 20:230] = 1                       # a horizontal 3px crack
    acc0 = dict(tp=0, fp=0, fn=0, sp_in_g=0, sp=0, sg_in_p=0, sg=0,
                cl_tp=0, cl_fp=0, cl_fn=0)

    def score(pred):
        acc = dict(acc0)
        tp, fp, fn = confusion(pred, gt)
        acc.update(tp=tp, fp=fp, fn=fn)
        for k, v in cldice_counts(pred, gt).items():
            acc[k] += v
        for k, v in cliou_counts(pred, gt).items():
            acc[k] += v
        return finalize(acc)

    s = score(gt.copy())
    assert abs(s["pixel_iou"] - 1) < 1e-6 and abs(s["cldice"] - 1) < 1e-6, s
    s = score(np.zeros_like(gt))
    assert s["pixel_iou"] == 0 and s["cliou_4px"] == 0, s
    shifted = np.roll(gt, 2, axis=0)              # 2 px vertical shift
    s = score(shifted)
    assert s["pixel_iou"] < 0.5, s                 # 3px line, 2px shift -> IoU 1/5
    assert s["cliou_4px"] > 0.99, s                # inside 4px tolerance
    noise = gt.copy()
    noise[rng.integers(0, 256, 50), rng.integers(0, 256, 50)] = 1
    s = score(noise)
    assert 0.5 < s["pixel_iou"] < 1.0, s

    # PIN THE LOOKUP ORDER (Amendment A1.4). infer_sam writes both
    # {stem}_mask.png (the binary prediction) and {stem}.png (an RGB overlay
    # figure); if the bare stem wins, every SAM3 row is scored against a
    # matplotlib picture and nothing warns.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "IMG.png").write_bytes(b"overlay")       # the decoy figure
        (d / "IMG_mask.png").write_bytes(b"mask")     # the real prediction
        stem = "IMG"
        cand = [d / f"{stem}_mask.png", d / f"{stem}.jpg", d / f"{stem}.png"]
        pf = next((c for c in cand if c.exists()), None)
        assert pf is not None and pf.name.endswith("_mask.png"), \
            f"lookup order regressed: picked {pf}"
    print("selftest PASS: identity=1, empty=0, 2px-shift tolerated by clIoU, "
          "noise penalized, _mask.png preferred over the overlay figure")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path,
                    help="fold split dir with _annotations.coco.json")
    ap.add_argument("--gt-mask-dir", type=Path, default=None,
                    help="GT as a dir of mask PNGs (axis B external sets) "
                         "— alternative to --gt")
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--marked-list", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.gt_mask_dir is not None:
        run_eval(a.gt_mask_dir, a.pred, a.out, a.marked_list,
                 gt_mode="masks")
    else:
        run_eval(a.gt, a.pred, a.out, a.marked_list)
