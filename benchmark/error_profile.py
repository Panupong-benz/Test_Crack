# -*- coding: utf-8 -*-
"""error_profile - WHAT KIND of error is left, per image (Amendment A1.23).

DESCRIPTIVE / EXPLORATORY, and declared as such before any mask existed:
it never enters main_table.csv, never enters a verdict, and is never used to
rank models. It is computed from the frozen masks and the frozen GT, so it
cannot move a number that has already been reported.

The reason it exists: a scalar cannot separate the failure modes that lead to
different fixes. cliou_4px = 0.6 is equally consistent with "the crack is
traced but too thin and breaks" and "the model fires on grid lines". Four
paired columns split them:

  fp_isolated_px  FP inside components that never touch GT  -> wrong object
                  (grid lines, written numbers)             -> SS7.2 negatives
  fp_touching_px  FP inside components that do touch GT     -> merely thick
                                                            -> threshold
  fn_broken_px    FN inside GT components that ARE partly   -> the crack breaks
                  detected                                  -> SS7.1 fusion
  fn_missed_px    FN inside GT components with no overlap   -> crack not seen
                                                            -> coverage/data
  n_cc_pred/n_cc_gt   fragmentation - the quantity SS8bd measured as the
                      binding constraint on linking (F1 0.62 vs 0.84 ceiling)

GT comes from eval_masks.load_gt_masks and the prediction is located by
render_overlays.find_pred - the same two single sources the metrics use.

Usage:
  python error_profile.py --results results/benchmark --data-root data \
      --runs-dir runs [--marked-list marked_line_images.txt]
  python error_profile.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_masks import load_gt_masks            # noqa: E402
from render_overlays import find_pred, run_tag  # noqa: E402

# mirrors summarize_benchmark.TAG_RE
TAG_RE = re.compile(r"^eval_([a-z0-9]+)_([A-Za-z0-9]+?)(?:_s(\d+))?$")

FIELDS = ["model", "fold", "seed", "image", "marked",
          "gt_px", "pred_px", "tp", "fp", "fn",
          "n_cc_gt", "n_cc_pred", "frag_ratio",
          "skel_len_gt", "skel_len_pred",
          "fp_isolated_px", "fp_touching_px",
          "fn_broken_px", "fn_missed_px"]


def _skel_len(m: np.ndarray) -> int:
    """Skeleton pixel count. skimage, never cv2.ximgproc - it is not
    installed here and its absence silently produced a non-skeleton once
    (SS8f: 52% of pixels had degree >= 3, inflating lengths 25-30%)."""
    if not m.any():
        return 0
    from skimage.morphology import skeletonize
    return int(skeletonize(m > 0).sum())


def profile_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pure function - selftested on planted masks."""
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    tp = int((p & g).sum())
    fp_m = (p == 1) & (g == 0)
    fn_m = (g == 1) & (p == 0)

    n_p, lab_p = cv2.connectedComponents(p, connectivity=8)
    n_g, lab_g = cv2.connectedComponents(g, connectivity=8)
    n_p, n_g = n_p - 1, n_g - 1        # drop the background label

    # a pred component is "isolated" if it shares no pixel with GT
    touch_p = sorted(set(np.unique(lab_p[g == 1]).tolist()) - {0})
    fp_iso = int(fp_m[~np.isin(lab_p, touch_p)].sum()) if n_p else 0
    fp_touch = int(fp_m.sum()) - fp_iso

    # a GT component is "detected" if it shares any pixel with the prediction
    det_g = sorted(set(np.unique(lab_g[p == 1]).tolist()) - {0})
    fn_missed = int(fn_m[~np.isin(lab_g, det_g)].sum()) if n_g else 0
    fn_broken = int(fn_m.sum()) - fn_missed

    return {"gt_px": int(g.sum()), "pred_px": int(p.sum()), "tp": tp,
            "fp": int(fp_m.sum()), "fn": int(fn_m.sum()),
            "n_cc_gt": n_g, "n_cc_pred": n_p,
            "frag_ratio": "" if n_g == 0 else round(n_p / n_g, 4),
            "skel_len_gt": _skel_len(g), "skel_len_pred": _skel_len(p),
            "fp_isolated_px": fp_iso, "fp_touching_px": fp_touch,
            "fn_broken_px": fn_broken, "fn_missed_px": fn_missed}


def discover(results: Path):
    """(model, fold, seed) triples from the eval CSVs actually present."""
    out = []
    for f in sorted(results.glob("eval_*.csv")):
        m = TAG_RE.match(f.stem)
        if m:
            out.append((m.group(1), m.group(2),
                        "" if m.group(3) is None else m.group(3)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--results", type=Path, default=Path("results/benchmark"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--marked-list", type=Path,
                    default=Path("marked_line_images.txt"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    marked = set()
    if a.marked_list.exists():
        marked = {ln.strip() for ln in
                  a.marked_list.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")}

    rows, gt_cache = [], {}
    triples = discover(a.results)
    if not triples:
        print(f"no eval_*.csv under {a.results} - nothing to profile")
        return 0
    for model, wall, seed in triples:
        fold = a.data_root / f"fold_{wall}"
        if wall not in gt_cache:
            gt_cache[wall] = load_gt_masks(fold / "test")
        gts = gt_cache[wall]
        pred_dir = a.runs_dir / run_tag(model, wall, seed or 0) / "masks"
        for name, gt in sorted(gts.items()):
            pf = find_pred(pred_dir, name)
            pred = (cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
                    if pf is not None else np.zeros_like(gt))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            r = {"model": model, "fold": wall, "seed": seed, "image": name,
                 "marked": int(name in marked or Path(name).stem in marked)}
            r.update(profile_pair(pred, gt))
            rows.append(r)
        print(f"profiled {model}/{wall}/{seed or '-'}: {len(gts)} images")

    out = a.out or (a.results / "error_profile.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"error_profile -> {out} ({len(rows)} rows) "
          f"[DESCRIPTIVE: not a verdict, not in main_table]")
    return 0


def selftest() -> int:
    """Planted masks with every class present exactly once, counted by hand."""
    gt = np.zeros((64, 64), np.uint8)
    pred = np.zeros((64, 64), np.uint8)
    gt[20, 10:51] = 1                 # crack A, 41 px
    gt[40, 10:21] = 1                 # crack B, 11 px - never detected
    pred[20, 10:30] = 1               # A, left piece  (20 px)
    pred[20, 35:51] = 1               # A, right piece (16 px), gap = 5 px
    pred[21, 15] = 1                  # 1 px of over-thickness, touches A
    pred[50:60, 50:60] = 1            # 100 px hallucination, touches nothing

    r = profile_pair(pred, gt)
    assert r["gt_px"] == 52 and r["pred_px"] == 137, r
    assert r["tp"] == 36 and r["fp"] == 101 and r["fn"] == 16, r
    assert r["fp_isolated_px"] == 100, r      # the blob only
    assert r["fp_touching_px"] == 1, r        # the over-thick pixel only
    assert r["fn_broken_px"] == 5, r          # the gap inside crack A
    assert r["fn_missed_px"] == 11, r         # all of crack B
    assert r["n_cc_gt"] == 2 and r["n_cc_pred"] == 3, r
    assert r["frag_ratio"] == 1.5, r
    assert r["skel_len_gt"] > 0 and r["skel_len_pred"] > 0, r

    empty = profile_pair(np.zeros((8, 8), np.uint8),
                         np.zeros((8, 8), np.uint8))
    assert empty["frag_ratio"] == "" and empty["n_cc_pred"] == 0, empty
    fp_only = profile_pair(np.ones((8, 8), np.uint8),
                           np.zeros((8, 8), np.uint8))
    assert fp_only["fp_isolated_px"] == 64 and fp_only["fp_touching_px"] == 0

    print("selftest PASS: fp_isolated=100 / fp_touching=1 / fn_broken=5 / "
          "fn_missed=11 / cc 3 vs 2 on planted masks; empty and FP-only "
          "edge cases handled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
