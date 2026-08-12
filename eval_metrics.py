"""Journal evaluation suite for SAM3-LoRA crack segmentation (Stage C, 2026-08).

Scores predictions from infer_fused.py against COCO ground truth and produces
everything a paper needs in one pass:

  * per-pixel metrics per image: PixelIoU, Dice/F1, precision, recall,
    clDice (hard-skeleton, Shit et al. 2021), FP pixel count
  * threshold sweep on the 'whole' .npz instances (valid split) ->
    threshold_sweep.csv + sweep curve PNG + the argmax-F1 threshold
  * fixed-threshold evaluation on the test split -> test_metrics.json
  * per-drift breakdown (drift parsed from coco_with_meta.csv) -> CSV + PNG
  * qualitative panels: image | GT overlay | prediction overlay
  * training curves from val_stats.json (loss per epoch)

Modes:
  --npz DIR   score re-thresholdable whole-image instance sets (from
              infer_fused --variants whole); --thresholds sweeps for free
  --masks DIR score binary PNG masks (tilemax / tilemean variants) at the
              resolution they were saved in

GT is rasterised from COCO polygons at the prediction's own resolution, so
whole (<=1008 px) and tiled (native res) variants are each scored
self-consistently. Only the 'Crack' category is evaluated.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


# ----------------------------------------------------------------- metrics

def skeletonize_bool(m):
    from skimage.morphology import skeletonize
    return skeletonize(m > 0)


def cl_dice(pred, gt):
    """clDice (hard): topology-aware score for thin structures."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp, sg = skeletonize_bool(pred), skeletonize_bool(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)      # topology precision
    tsens = (sg & pred).sum() / max(sg.sum(), 1)    # topology sensitivity
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def pixel_metrics(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return dict(iou=iou, dice=dice, precision=prec, recall=rec,
                tp=tp, fp=fp, fn=fn)


# ----------------------------------------------------------------- data

def load_gt(data_dir, crack_only=True):
    """image stem -> (list of polygon anns, (H, W))."""
    with open(Path(data_dir) / "_annotations.coco.json") as f:
        coco = json.load(f)
    crack_ids = {c["id"] for c in coco["categories"]
                 if (not crack_only) or c["name"].lower() == "crack"}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in crack_ids:
            anns[a["image_id"]].append(a)
    out = {}
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        out[stem] = (anns.get(im["id"], []), (im["height"], im["width"]),
                     im["file_name"])
    return out


def gt_mask(ann_list, hw, out_hw):
    """Rasterise polygons at out_hw (scaled from native hw)."""
    import cv2
    H, W = hw
    h, w = out_hw
    m = np.zeros((h, w), np.uint8)
    sx, sy = w / W, h / H
    for a in ann_list:
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list):
            continue
        for poly in seg:
            p = np.asarray(poly, np.float64).reshape(-1, 2)
            p[:, 0] *= sx
            p[:, 1] *= sy
            cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m.astype(bool)


def load_npz(path):
    z = np.load(path)
    shape = tuple(z["mask_shape"])
    if shape[0] == 0:
        masks = np.zeros(shape, bool)
    else:
        masks = np.unpackbits(z["masks"])[: int(np.prod(shape))] \
            .reshape(shape).astype(bool)
    return z["scores"], masks, tuple(z["orig_size"])


def drift_of(stem, meta_rows):
    r = meta_rows.get(stem.split(".rf")[0].split("_jpg")[0]) or meta_rows.get(stem)
    if r is None:
        for k, v in meta_rows.items():
            if k in stem:
                return v
        return None
    return r


def load_meta(meta_csv):
    """stem / img_core -> |drift| from coco_with_meta.csv
    (columns: coco_file_name,img_core,wall,drift,...)."""
    rows = {}
    if not meta_csv or not Path(meta_csv).exists():
        return rows
    with open(meta_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                d = abs(float(r.get("drift_num") or r.get("drift")))
            except (TypeError, ValueError):
                continue
            if r.get("coco_file_name"):
                rows[Path(r["coco_file_name"]).stem] = d
            if r.get("img_core"):
                rows[r["img_core"]] = d
    return rows


# ----------------------------------------------------------------- scoring

def score_split(pred_iter, out_dir, tag, meta_rows, panels_from=None,
                n_panels=12):
    """pred_iter yields (stem, pred_bool, gt_bool, native_file_or_None)."""
    rows = []
    panel_pool = []
    for stem, pred, gt, img_path in pred_iter:
        m = pixel_metrics(pred, gt)
        m["cldice"] = cl_dice(pred, gt)
        m["stem"] = stem
        d = drift_of(stem, meta_rows)
        m["drift"] = d if d is not None else ""
        rows.append(m)
        if img_path is not None:
            panel_pool.append((stem, img_path, pred, gt, d))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"per_image_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        if not sel:
            return {}
        return {k: float(np.mean([r[k] for r in sel]))
                for k in ("iou", "dice", "precision", "recall", "cldice")}

    summary = dict(n_images=len(rows), mean=agg(rows))
    # micro (pixel-pooled) IoU/Dice - robust to image-size imbalance
    TP = sum(r["tp"] for r in rows)
    FP = sum(r["fp"] for r in rows)
    FN = sum(r["fn"] for r in rows)
    summary["micro"] = dict(
        iou=TP / (TP + FP + FN) if TP + FP + FN else 1.0,
        dice=2 * TP / (2 * TP + FP + FN) if TP + FP + FN else 1.0)
    # per-drift
    by_d = defaultdict(list)
    for r in rows:
        if r["drift"] != "":
            by_d[r["drift"]].append(r)
    summary["per_drift"] = {str(k): dict(n=len(v), **agg(v))
                            for k, v in sorted(by_d.items())}
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if panel_pool and n_panels:
        render_panels(panel_pool, out_dir / f"panels_{tag}", n_panels)
    return summary


def render_panels(pool, panel_dir, n):
    import cv2
    panel_dir.mkdir(parents=True, exist_ok=True)
    pool = sorted(pool, key=lambda t: (t[4] if t[4] is not None else -1, t[0]))
    step = max(1, len(pool) // n)
    for stem, img_path, pred, gt, d in pool[::step][:n]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = pred.shape
        img = cv2.resize(img, (w, h))
        gt_o, pr_o = img.copy(), img.copy()
        gt_o[gt] = (0.4 * gt_o[gt] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)
        pr_o[pred] = (0.4 * pr_o[pred] + 0.6 * np.array([0, 0, 220])).astype(np.uint8)
        panel = np.concatenate([img, gt_o, pr_o], axis=1)
        label = f"{stem}  drift={d}  (mid=GT green, right=pred red)"
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(panel_dir / f"{stem}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------- plots

def plot_sweep(sweep_rows, out_png, best_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["threshold"] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, style in (("dice", "-o"), ("iou", "-s"), ("cldice", "-^")):
        ax.plot(t, [r[k] for r in sweep_rows], style, label=k, ms=4)
    ax.axvline(best_t, ls="--", c="gray", label=f"chosen t*={best_t}")
    ax.set_xlabel("detection score threshold")
    ax.set_ylabel("metric (valid split)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def plot_curves(val_stats_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep, tr, va = [], [], []
    with open(val_stats_path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep.append(j.get("epoch"))
            tr.append(j.get("train_loss"))
            va.append(j.get("val_loss"))
    if not ep:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, tr, "-o", label="train loss", ms=4)
    ax.plot(ep, va, "-s", label="val loss", ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def plot_per_drift(summary, out_png, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = summary.get("per_drift", {})
    if not pd:
        return
    d = sorted(pd, key=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, style in (("dice", "-o"), ("iou", "-s"), ("cldice", "-^")):
        ax.plot([float(x) for x in d], [pd[x][k] for x in d], style,
                label=k, ms=4)
    ax.set_xlabel("drift level (%)")
    ax.set_ylabel(f"metric ({tag})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="split dir with COCO json")
    ap.add_argument("--npz", help="dir of whole-image instance .npz")
    ap.add_argument("--masks", help="dir of binary PNG masks")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed score threshold (npz mode)")
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help="sweep grid (npz mode when --threshold not given)")
    ap.add_argument("--meta_csv", default="coco_with_meta.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--val_stats", help="also plot training curves from this file")
    args = ap.parse_args()
    if not args.npz and not args.masks:
        ap.error("need --npz or --masks")

    gt = load_gt(args.data_dir)
    meta = load_meta(args.meta_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    if args.val_stats and Path(args.val_stats).exists():
        plot_curves(args.val_stats, out / "training_curves.png")

    if args.masks:
        def it():
            for p in sorted(Path(args.masks).glob("*.png")):
                if p.stem not in gt:
                    continue
                pred = np.asarray(PILImage.open(p)) > 127
                anns, hw, fname = gt[p.stem]
                yield p.stem, pred, gt_mask(anns, hw, pred.shape), data / fname
        s = score_split(it(), out, args.tag, meta, n_panels=args.panels)
        plot_per_drift(s, out / f"per_drift_{args.tag}.png", args.tag)
        print(json.dumps({args.tag: s["mean"]}, indent=2))
        return

    # npz mode: cache unions per threshold from the instance sets
    files = sorted(Path(args.npz).glob("*.npz"))
    files = [p for p in files if p.stem in gt]

    def union_at(scores, masks, t):
        keep = scores >= t
        if not keep.any():
            return np.zeros(masks.shape[1:] if masks.ndim == 3 else (1, 1), bool)
        return np.any(masks[keep], axis=0)

    if args.threshold is None:
        sweep = []
        for t in args.thresholds:
            accs = defaultdict(list)
            for p in files:
                scores, masks, _orig = load_npz(p)
                pred = union_at(scores, masks, t)
                anns, hw, _f = gt[p.stem]
                g = gt_mask(anns, hw, pred.shape)
                m = pixel_metrics(pred, g)
                m["cldice"] = cl_dice(pred, g)
                for k in ("iou", "dice", "cldice"):
                    accs[k].append(m[k])
            sweep.append(dict(threshold=t,
                              **{k: float(np.mean(v)) for k, v in accs.items()}))
            print(f"t={t:.2f}  dice={sweep[-1]['dice']:.4f}  "
                  f"iou={sweep[-1]['iou']:.4f}  cldice={sweep[-1]['cldice']:.4f}")
        best = max(sweep, key=lambda r: r["dice"])
        with open(out / "threshold_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)
        plot_sweep(sweep, out / "threshold_sweep.png", best["threshold"])
        with open(out / "chosen_threshold.json", "w") as f:
            json.dump(best, f, indent=2)
        print(f"chosen t* = {best['threshold']} (max mean Dice on this split)")
        return

    t = args.threshold

    def it():
        for p in files:
            scores, masks, _orig = load_npz(p)
            pred = union_at(scores, masks, t)
            anns, hw, fname = gt[p.stem]
            yield p.stem, pred, gt_mask(anns, hw, pred.shape), data / fname
    s = score_split(it(), out, args.tag, meta, n_panels=args.panels)
    plot_per_drift(s, out / f"per_drift_{args.tag}.png", args.tag)
    print(json.dumps({args.tag: {"threshold": t, **s["mean"],
                                 "micro": s["micro"]}}, indent=2))


if __name__ == "__main__":
    main()
