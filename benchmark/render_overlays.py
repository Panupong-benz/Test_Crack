# -*- coding: utf-8 -*-
"""FP/FN overlay renderer + qualitative model-comparison panels
(benchmark_protocol.md Amendment A1.2: overlay = TP green / FP red /
FN blue on the photo; panels = same crop across all model rows).

Three jobs, one color code:
  1) Error analysis (model improvement): --worst N reads
     per_image_metrics.csv, picks each model's N lowest-scoring test
     images, renders their overlays -> the most direct starting point
     for "why did this model fail here".
  2) Paper qualitative figure: --panel renders one row per image x one
     column per model (crop via --crop x,y,w,h), for the 4 declared crop
     types (clear crack / grid line / written number / dense zone).
  3) Label-vs-prediction comparison (A1.23): `compare` writes the 4-panel
     figure - photo | our label | model mask | TP/FP/FN - AND the four
     panels as separate PNGs, so a different figure can be built from the
     same pixels. `figset` executes the FROZEN selection rules (A1.23 items
     124-125) and logs every emitted figure to figset_manifest.csv: no
     image and no crop is ever chosen by eye. Figures diagnose and explain;
     they never decide a ranking (item 123).

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
  python render_overlays.py compare --fold data/fold_RW20 --model a6       --image IMG_4128.jpg --runs-dir runs --out figs/compare       --images-dir ../../03_annotation/_pool/POOL_BM/train
  python render_overlays.py figset --fold data/fold_RW20 --models a6 a5       --per-image results/benchmark/per_image_metrics.csv       --marked-list marked_line_images.txt --runs-dir runs --out figs/figset
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
# panel (b) label colour = the annotation tool's own amber
# (03_annotation/crop_wall.py ANNO_COLOR/ANNO_ALPHA), so the "our label" panel
# looks like what the labeller saw; panel (c) prediction gets its own hue.
C_GT, A_GT = (0, 200, 255), 0.45
C_PRED, A_PRED = (230, 120, 60), 0.45
ALPHA = 0.65
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# same set as cmd_panel / summarize_benchmark.MODELS - a5/a1 carry no seed
SEEDLESS = {"a1", "a5"}
# A1.23 item 125: tile-sized window, quarter-tile stride, ties by (y, x)
WIN, STRIDE = 1008, 252


def find_pred(pred_dir: Path, name: str):
    """Candidate order MUST match eval_masks (A1.4/A1.23 item 129).

    infer_sam writes BOTH {stem}_mask.png (the binary prediction) and
    {stem}.png (its own matplotlib overlay figure) into the same dir, so a
    reversed order silently paints the decoy figure as the prediction.
    """
    stem = Path(name).stem
    for c in (pred_dir / f"{stem}_mask.png", pred_dir / name,
              pred_dir / f"{stem}.png"):
        if c.exists():
            return c
    return None


def run_tag(model: str, wall: str, seed) -> str:
    return f"{model}_{wall}" + ("" if model in SEEDLESS else f"_s{seed}")


def dilate(m: np.ndarray, px: int) -> np.ndarray:
    """Display-only thickening (A1.23 item 128) - applied identically to GT
    and prediction, never to anything that is counted."""
    if px <= 0:
        return m
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate((m > 0).astype(np.uint8), k)


def paint_mask(img: np.ndarray, mask: np.ndarray, color, alpha) -> np.ndarray:
    """Paint ONE mask over the photo (panels b and c). Pure function."""
    out = img.copy()
    m = mask > 0
    out[m] = (alpha * np.array(color) + (1 - alpha) * out[m]).astype(np.uint8)
    return out


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


def load_pair(fold: Path, pred_dir: Path, name: str, gts: dict,
              images_dir: Path = None):
    """images_dir: fallback source of the photo. The fold dirs are rebuilt on
    the instance but not materialised locally (the dry runs used --no-images),
    so local rendering reads the photos straight out of the pool."""
    img_p = fold / "test" / name
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    if img is None and images_dir is not None:
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_COLOR)
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
        img, pred, gt = load_pair(args.fold, args.pred, name, gts,
                                  args.images_dir)
        if img is None:
            print(f"SKIP unreadable {name}")
            continue
        d = args.display_dilate
        cv2.imwrite(str(args.out / f"{Path(name).stem}_overlay.png"),
                    overlay_image(img, dilate(pred, d), dilate(gt, d)))
        print(f"overlay {name}")


def cmd_worst(args):
    rows = [r for r in csv.DictReader(open(args.per_image, encoding="utf-8"))
            if r["model"] == args.model and str(r["seed"]) == str(args.seed)
            and r["fold"] == args.fold.name.replace("fold_", "")]
    rows.sort(key=lambda r: float(r[args.metric]))
    picks = rows[:args.n]
    pred_dir = args.runs_dir / run_tag(
        args.model, args.fold.name.replace("fold_", ""), args.seed) / "masks"
    gts = load_gt_masks(args.fold / "test")
    args.out.mkdir(parents=True, exist_ok=True)
    for r in picks:
        img, pred, gt = load_pair(args.fold, pred_dir, r["image"], gts,
                                  args.images_dir)
        if img is None:
            continue
        d = args.display_dilate
        out = overlay_image(img, dilate(pred, d), dilate(gt, d))
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
        tag = run_tag(model, fold_wall, args.seed)
        img, pred, gt = load_pair(args.fold, args.runs_dir / tag / "masks",
                                  args.image, gts, args.images_dir)
        if img is None:
            print(f"SKIP {model}: image unreadable")
            continue
        d = args.display_dilate
        ov = overlay_image(img, dilate(pred, d), dilate(gt, d))
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


# --------------------------------------------------------------------------
# A1.23: label-vs-prediction comparison (4 panels) + the raw panels beside it
# --------------------------------------------------------------------------
PANEL_LABELS = ("(a) photo", "(b) our label", "(c) model mask",
                "(d) TP/FP/FN")


def best_window(mask: np.ndarray, win: int = WIN, stride: int = STRIDE):
    """Window of size win maximising mask sum; ties resolved by (y, x) min.

    Deterministic by construction - this is what makes the A1.23 item 125
    mechanism crops a rule rather than a choice."""
    h, w = mask.shape[:2]
    win = min(win, h, w)
    integ = cv2.integral((mask > 0).astype(np.uint8))
    best, bxy = -1, (0, 0)
    for y in range(0, max(h - win, 0) + 1, stride):
        for x in range(0, max(w - win, 0) + 1, stride):
            v = int(integ[y + win, x + win] - integ[y, x + win]
                    - integ[y + win, x] + integ[y, x])
            if v > best:
                best, bxy = v, (x, y)
    return bxy[0], bxy[1], win, win


def _crop(a, box):
    if box is None:
        return a
    x, y, w, h = box
    return a[y:y + h, x:x + w]


def compare_panels(img, pred, gt, dilate_px=2, box=None):
    """The four panel images, in order. Dilation is display-only and is
    applied identically to GT and prediction (A1.23 item 128)."""
    g, pr = dilate(gt, dilate_px), dilate(pred, dilate_px)
    return [_crop(img, box),
            _crop(paint_mask(img, g, C_GT, A_GT), box),
            _crop(paint_mask(img, pr, C_PRED, A_PRED), box),
            _crop(overlay_image(img, pr, g), box)]


def compose_figure(panels, out_png, title="", dpi=300):
    """Composite figure, laid out like code/make_fig_fused_canvas.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]
    hh = min(pn.shape[0] for pn in panels)
    ws = [pn.shape[1] * hh / pn.shape[0] for pn in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(sum(ws) / hh * 3.2, 3.6),
                             gridspec_kw={"width_ratios": ws})
    for ax, pn, lab in zip(np.atleast_1d(axes), panels, PANEL_LABELS):
        ax.imshow(cv2.cvtColor(pn, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#B9B9B4")
            sp.set_linewidth(0.6)
        ax.set_xlabel(lab, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_compare(fold, pred_dir, name, gts, out_dir, stem_prefix,
                   images_dir=None, dilate_px=2, box=None, title="", dpi=300):
    """Write the composite AND the four panels as separate files - the raw
    material the figure is made of, so a different figure can be built from
    the same pixels without re-running anything."""
    img, pred, gt = load_pair(fold, pred_dir, name, gts, images_dir)
    if img is None:
        return None
    panels = compare_panels(img, pred, gt, dilate_px, box)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pn, sfx in zip(panels, ("a_photo", "b_label", "c_pred", "d_agree")):
        cv2.imwrite(str(out_dir / f"{stem_prefix}__{sfx}.png"), pn)
    fig = out_dir / f"fig_compare_{stem_prefix}.png"
    compose_figure(panels, fig, title, dpi)
    return fig


def cmd_compare(args):
    gts = load_gt_masks(args.fold / "test")
    wall = args.fold.name.replace("fold_", "")
    pred_dir = args.runs_dir / run_tag(args.model, wall, args.seed) / "masks"
    box = tuple(map(int, args.crop.split(","))) if args.crop else None
    stem = Path(args.image).stem
    f = render_compare(args.fold, pred_dir, args.image, gts, args.out,
                       f"{args.model}_{wall}_{stem}", args.images_dir,
                       args.display_dilate, box,
                       args.title or f"{args.model} / {wall} / {stem}",
                       args.dpi)
    if f is None:
        print(f"SKIP unreadable {args.image}")
        return 1
    print(f"compare -> {f}")
    return 0


# --------------------------------------------------------------------------
# A1.23 items 124-125: the FROZEN figure set. No image and no crop is chosen
# by eye - every row of figset_manifest.csv names the rule that produced it,
# and a figure without a manifest row may not appear in the paper.
# --------------------------------------------------------------------------
MANIFEST_FIELDS = ["rule", "model", "fold", "seed", "image", "crop",
                   "metric", "value", "figure"]


def seed_col(model: str, seed) -> str:
    """summarize_benchmark writes "" in the seed column for seedless rows."""
    return "" if model in SEEDLESS else str(seed)


def _rows(per_image: Path, model: str, wall: str, seed) -> list:
    sc = seed_col(model, seed)
    with open(per_image, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["model"] == model and r["fold"] == wall
                and str(r["seed"]) == sc]
    return rows


def pick_data_driven(rows: list, metric: str = "cliou_4px") -> list:
    """worst / median / best - sorted ascending, ties by image name."""
    rs = sorted(rows, key=lambda r: (float(r[metric]), r["image"]))
    if not rs:
        return []
    return [("worst", rs[0]), ("median", rs[len(rs) // 2]), ("best", rs[-1])]


def pick_mechanism(rows: list, marked: set) -> list:
    """The four declared crops. `written_number` uses the empty-GT rows,
    which are by construction exactly initial_fp.csv's row set."""
    def gtpx(r):
        return float(r["tp"]) + float(r["fn"])

    out = []
    withgt = [r for r in rows if gtpx(r) > 0]
    empty = [r for r in rows if gtpx(r) == 0]
    mk = [r for r in withgt if r["image"] in marked
          or Path(r["image"]).stem in marked]
    if withgt:
        out.append(("clear_crack", max(
            withgt, key=lambda r: (float(r["cliou_4px"]), r["image"])), "gt"))
    if mk:
        out.append(("grid_line", max(
            mk, key=lambda r: (float(r["fp"]), r["image"])), "fp"))
    if empty:
        out.append(("written_number", max(
            empty, key=lambda r: (float(r["fp"]), r["image"])), "fp"))
    if withgt:
        out.append(("dense_zone", max(
            withgt, key=lambda r: (gtpx(r), r["image"])), "gt"))
    return out


def cmd_figset(args):
    gts = load_gt_masks(args.fold / "test")
    wall = args.fold.name.replace("fold_", "")
    # missing list = ERROR: the grid_line/written_number mechanism rules
    # would silently pick from an empty candidate set otherwise
    if not (args.marked_list and args.marked_list.exists()):
        raise SystemExit(f"FATAL: marked list not found: {args.marked_list}")
    marked = {l.strip() for l in
              args.marked_list.read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")}
    args.out.mkdir(parents=True, exist_ok=True)
    man = []
    for model in args.models:
        rows = _rows(args.per_image, model, wall, args.seed)
        if not rows:
            print(f"WARN no per-image rows for {model}/{wall}")
            continue
        pred_dir = args.runs_dir / run_tag(model, wall, args.seed) / "masks"
        for rule, r in pick_data_driven(rows, args.metric):
            stem = Path(r["image"]).stem
            f = render_compare(
                args.fold, pred_dir, r["image"], gts, args.out / "data_driven",
                f"{rule}_{model}_{wall}_{stem}", args.images_dir,
                args.display_dilate, None,
                f"{rule} | {model} | {wall} | {args.metric}="
                f"{float(r[args.metric]):.3f}", args.dpi)
            man.append({"rule": f"data_driven:{rule}", "model": model,
                        "fold": wall, "seed": seed_col(model, args.seed),
                        "image": r["image"], "crop": "",
                        "metric": args.metric, "value": r[args.metric],
                        "figure": "" if f is None else f.name})
        for rule, r, basis in pick_mechanism(rows, marked):
            img, pred, gt = load_pair(args.fold, pred_dir, r["image"], gts,
                                      args.images_dir)
            if img is None:
                print(f"SKIP unreadable {r['image']}")
                continue
            src = gt if basis == "gt" else ((pred > 0) & (gt == 0))
            box = best_window(src.astype(np.uint8))
            stem = Path(r["image"]).stem
            f = render_compare(
                args.fold, pred_dir, r["image"], gts, args.out / "mechanism",
                f"{rule}_{model}_{wall}_{stem}", args.images_dir,
                args.display_dilate, box,
                f"{rule} | {model} | {wall}", args.dpi)
            man.append({"rule": f"mechanism:{rule}", "model": model,
                        "fold": wall, "seed": seed_col(model, args.seed),
                        "image": r["image"],
                        "crop": ",".join(str(v) for v in box),
                        "metric": basis + "_px_in_window",
                        "value": int((src[box[1]:box[1] + box[3],
                                          box[0]:box[0] + box[2]] > 0).sum()),
                        "figure": "" if f is None else f.name})
    mpath = args.manifest or (args.out / "figset_manifest.csv")
    newfile = not mpath.exists()
    with open(mpath, "a" if args.append else "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if newfile or not args.append:
            w.writeheader()
        w.writerows(man)
    print(f"figset -> {args.out}  ({len(man)} figures)  manifest {mpath}")
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

    # A1.23 item 129: infer_sam writes BOTH {stem}_mask.png (the prediction)
    # and {stem}.png (its own overlay figure) into the same dir. The decoy
    # must never win, or the renderer paints the figure as the prediction.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cv2.imwrite(str(d / "IMG_1.png"), np.full((8, 8, 3), 255, np.uint8))
        cv2.imwrite(str(d / "IMG_1_mask.png"), np.zeros((8, 8), np.uint8))
        got = find_pred(d, "IMG_1.jpg")
        assert got is not None and got.name == "IMG_1_mask.png", got
        (d / "IMG_1_mask.png").unlink()
        assert find_pred(d, "IMG_1.jpg").name == "IMG_1.png"

    # display dilation is symmetric and never touches a counted quantity
    m = np.zeros((16, 16), np.uint8)
    m[8, 8] = 1
    assert int(dilate(m, 0).sum()) == 1
    assert int(dilate(m, 2).sum()) == 25

    # best_window: deterministic, ties to the smallest (y, x)
    big = np.zeros((300, 300), np.uint8)
    big[200:260, 200:260] = 1
    assert best_window(big, win=100, stride=50) == (200, 200, 100, 100)
    flat = np.ones((300, 300), np.uint8)
    assert best_window(flat, win=100, stride=50) == (0, 0, 100, 100)
    assert run_tag("a6", "RW20", 0) == "a6_RW20_s0"
    assert run_tag("a5", "RW20", 0) == "a5_RW20"

    print("selftest PASS: TP/FP/FN painted-pixel counts exact (50/50/50), "
          "background untouched; _mask.png beats the decoy overlay; "
          "dilation symmetric; best_window deterministic")
    return 0


def _common(sp):
    """Shared by every subcommand: where the photos are, and how thick the
    strokes are drawn (display only, identical for GT and prediction)."""
    sp.add_argument("--images-dir", type=Path, default=None,
                    help="fallback source of the photos when data/fold_*/test "
                         "is not materialised (e.g. _pool/POOL_BM/train)")
    sp.add_argument("--display-dilate", type=int, default=2,
                    help="A1.23 item 128: display-only dilation in px")
    sp.add_argument("--dpi", type=int, default=300)
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    o = _common(sub.add_parser("overlay"))
    o.add_argument("--fold", type=Path, required=True)
    o.add_argument("--pred", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--images", nargs="+", default=None)

    w = _common(sub.add_parser("worst"))
    w.add_argument("--fold", type=Path, required=True)
    w.add_argument("--per-image", type=Path, required=True)
    w.add_argument("--model", required=True)
    w.add_argument("--seed", default="0")
    w.add_argument("-n", type=int, default=5)
    w.add_argument("--metric", default="cliou_4px")
    w.add_argument("--runs-dir", type=Path, default=Path("runs"))
    w.add_argument("--out", type=Path, required=True)

    p = _common(sub.add_parser("panel"))
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seed", default="0")
    p.add_argument("--crop", default=None, help="x,y,w,h in image px")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, required=True)

    c = _common(sub.add_parser("compare"))
    c.add_argument("--fold", type=Path, required=True)
    c.add_argument("--image", required=True)
    c.add_argument("--model", default="a6")
    c.add_argument("--seed", default="0")
    c.add_argument("--crop", default=None, help="x,y,w,h in image px")
    c.add_argument("--runs-dir", type=Path, default=Path("runs"))
    c.add_argument("--title", default="")
    c.add_argument("--out", type=Path, required=True)

    g = _common(sub.add_parser("figset"))
    g.add_argument("--fold", type=Path, required=True)
    g.add_argument("--models", nargs="+", default=["a6", "a5"])
    g.add_argument("--seed", default="0")
    g.add_argument("--per-image", type=Path,
                   default=Path("results/benchmark/per_image_metrics.csv"))
    g.add_argument("--marked-list", type=Path,
                   default=Path(__file__).resolve().parent /
                   "marked_line_images.txt")
    g.add_argument("--metric", default="cliou_4px")
    g.add_argument("--runs-dir", type=Path, default=Path("runs"))
    g.add_argument("--manifest", type=Path, default=None)
    g.add_argument("--append", action="store_true",
                   help="append to an existing manifest (multi-fold runs)")
    g.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "overlay":
        return cmd_overlay(args)
    if args.cmd == "worst":
        return cmd_worst(args)
    if args.cmd == "panel":
        return cmd_panel(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "figset":
        return cmd_figset(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
