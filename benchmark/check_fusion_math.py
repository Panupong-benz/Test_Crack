# -*- coding: utf-8 -*-
"""Static gate for the overlap-fusion accumulation (docs/fusion_ab_spec.md G-F2).

WHY THIS FILE EXISTS. `infer_sam.py` imports `sam3` at module scope, so it
cannot be imported on the dev box — and by the rule recorded in 8bz, a file
that cannot be imported here MUST have a static gate, or its code is tested
for the first time with GPU money. The fusion math therefore lives in three
PURE-NUMPY module-level helpers (`_fusion_visit`, `_fusion_tile_accumulate`,
`_fusion_finalize`); this gate extracts them from the SHIPPED source with
`ast` and execs them with numpy only, so it runs the shipped bytes and cannot
drift from a copy.

WHAT IT CHECKS.
  1. A planted 3-tile scenario (two detecting tiles, one zero-detection
     visit, one edge tile clipped by the A1.32 rule) must reproduce a
     brute-force per-pixel reference computed here with independent loop
     arithmetic — max and mean, including the visit-count denominator.
  2. NEGATIVE case: an embedded BROKEN variant of the mean (divides by the
     number of DETECTING tiles instead of tile VISITS — the upward-bias
     mistake the spec's frozen wording exists to forbid) must produce a
     verdict that DIFFERS at the declared discriminator pixels. That is
     what proves this gate can detect the bias rather than passing always.
"""
import argparse
import ast
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "infer_sam.py",           # Test_Crack layout
              HERE.parent / "code" / "infer_sam.py"]  # crack-tool layout

FUNCS = ("_fusion_visit", "_fusion_tile_accumulate", "_fusion_finalize")

# The forbidden implementation: mean over DETECTING tiles. On the planted
# scenario it flips at least two pixels from False to True (the bias is
# always upward), which the negative case asserts.
BROKEN_SRC = textwrap.dedent("""
    def _fusion_finalize_broken(state, det_count, threshold):
        out = {}
        if "score_sum" in state:
            out["mean"] = (state["score_sum"] / np.maximum(det_count, 1)) > threshold
        return out
""")


def find_infer_sam(explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"check_fusion_math: {p} not found")
        return p
    for c in CANDIDATES:
        if c.exists():
            return c
    sys.exit("check_fusion_math: infer_sam.py not found in either layout")


def extract_funcs(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    scope = {"np": np}
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            seg = ast.get_source_segment(src, node)
            exec(textwrap.dedent(seg), scope)  # shipped bytes, not a copy
            found.append(node.name)
    missing = [f for f in FUNCS if f not in found]
    if missing:
        sys.exit(f"check_fusion_math: FAIL - {path} is missing module-level "
                 f"helper(s) {missing}. The fusion math must stay in these "
                 f"pure-numpy module-level functions so this gate can test "
                 f"the shipped source (fusion_ab_spec G-F2).")
    return scope


# ---- Planted scenario -------------------------------------------------
# Canvas 8x8, tile 4x4.
#   T1 at (0,0): detecting, uniform prob 0.6
#   T2 at (2,2): detecting, uniform prob 0.8
#   T3 at (4,4): VISIT ONLY (zero detections -> no accumulate call)
#   T4 at (6,6): detecting, uniform prob 0.9, PADDED past the canvas
#                (arrives as a full 4x4 prob map; the A1.32 clip must drop
#                the padded rows/cols)
H = W = 8
TILE = 4
TILES = [  # (xo, yo, prob_value or None for visit-only)
    (0, 0, 0.6),
    (2, 2, 0.8),
    (4, 4, None),
    (6, 6, 0.9),
]
THR = 0.5

# Pixels where the broken mean MUST disagree with the correct mean:
#   (4,4): sum=0.8, visits=2 (T2+T3) -> 0.4 False | detecting=1 -> 0.8 True
#   (6,6): sum=0.9, visits=2 (T3+T4) -> 0.45 False | detecting=1 -> 0.9 True
DISCRIMINATORS = [(4, 4), (6, 6)]


def run_extracted(scope):
    state = {"score_max": np.zeros((H, W), np.float32),
             "score_sum": np.zeros((H, W), np.float32)}
    count = np.zeros((H, W), np.uint16)
    det_count = np.zeros((H, W), np.uint16)  # for the broken variant only
    for xo, yo, p in TILES:
        scope["_fusion_visit"](count, xo, yo, TILE, TILE, H, W)
        if p is not None:
            prob = np.full((TILE, TILE), p, np.float32)
            scope["_fusion_tile_accumulate"](state, prob, xo, yo, H, W)
            th = min(TILE, H - yo); tw = min(TILE, W - xo)
            det_count[yo:yo + th, xo:xo + tw] += 1
    fused = scope["_fusion_finalize"](state, count, THR)
    return state, count, det_count, fused


def reference():
    """Brute-force per-pixel loops — independent arithmetic, no slicing."""
    smax = np.zeros((H, W), np.float32)
    ssum = np.zeros((H, W), np.float32)
    visits = np.zeros((H, W), np.int64)
    for xo, yo, p in TILES:
        for dy in range(TILE):
            for dx in range(TILE):
                y, x = yo + dy, xo + dx
                if y >= H or x >= W:
                    continue  # the padded region is not part of the image
                visits[y, x] += 1
                if p is not None:
                    smax[y, x] = max(smax[y, x], p)
                    ssum[y, x] += p
    ref_max = smax > THR
    ref_mean = np.zeros((H, W), bool)
    for y in range(H):
        for x in range(W):
            if visits[y, x] > 0:
                ref_mean[y, x] = (ssum[y, x] / visits[y, x]) > THR
    return ref_max, ref_mean, visits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="explicit path to infer_sam.py (default: search "
                         "both layouts)")
    args = ap.parse_args()

    path = find_infer_sam(args.file)
    scope = extract_funcs(path)

    state, count, det_count, fused = run_extracted(scope)
    ref_max, ref_mean, ref_visits = reference()

    # 1) visit counting (incl. zero-detection tiles + edge clipping)
    if not np.array_equal(count.astype(np.int64), ref_visits):
        sys.exit("check_fusion_math: FAIL - visit count differs from the "
                 "brute-force reference (zero-detection tiles or the A1.32 "
                 "clip are being miscounted)")
    print(f"  [ok] visits      {int(ref_visits.max())} max visits, "
          f"zero-detection tile counted, padded rows/cols dropped")

    # 2) fused max
    if not np.array_equal(fused["max"], ref_max):
        sys.exit("check_fusion_math: FAIL - fused max mask differs from "
                 "reference")
    print(f"  [ok] max         {int(ref_max.sum())} px above threshold, "
          f"matches reference exactly")

    # 3) fused mean (the visit-count denominator)
    if not np.array_equal(fused["mean"], ref_mean):
        sys.exit("check_fusion_math: FAIL - fused mean mask differs from "
                 "reference (denominator must be TILE VISITS, spec 2)")
    print(f"  [ok] mean        {int(ref_mean.sum())} px above threshold, "
          f"visit-count denominator confirmed")

    # 4) NEGATIVE: the detecting-tiles denominator must be DETECTED as
    #    different — otherwise this gate proves nothing.
    bscope = {"np": np}
    exec(BROKEN_SRC, bscope)
    broken = bscope["_fusion_finalize_broken"](state, det_count, THR)
    flips = [(y, x) for (y, x) in DISCRIMINATORS
             if bool(broken["mean"][y, x]) != bool(ref_mean[y, x])]
    if len(flips) != len(DISCRIMINATORS):
        sys.exit("check_fusion_math: FAIL - the broken (detecting-tiles) "
                 "mean did NOT flip the discriminator pixels; the planted "
                 "scenario no longer separates the two denominators and "
                 "this gate has gone blind")
    print(f"  [ok] negative    broken denominator flips "
          f"{len(flips)}/{len(DISCRIMINATORS)} discriminator pixels "
          f"(upward bias detected)")

    print(f"check_fusion_math: PASS ({path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
