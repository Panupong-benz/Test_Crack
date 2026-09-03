# -*- coding: utf-8 -*-
"""Static gate for the sliding-window mask paste (Amendment A1.32).

WHY THIS FILE EXISTS. `infer_sam.py` imports `sam3` at module scope, so it
cannot be imported on the dev box - and by the rule recorded in 8bz, a file
that cannot be imported here MUST have a static gate, or its code is tested
for the first time with GPU money. That is exactly what happened on
2026-09-03: `predict_sliding_window` had never run on an image with one side
below tile_size, and the first one it met stopped a paid queue after 16.34 h
of training with

    ValueError: operands could not be broadcast together with
                shapes (1008,951) (1008,1008) (1008,951)

PIL's crop() PADS past the image edge, so a window that runs off the canvas
comes back at the full tile_size while the destination slice is short. The
both-sides-short case takes the single-image fallback, so only the
exactly-one-side-short case ever reaches the paste.

HOW IT TESTS THE REAL CODE WITHOUT IMPORTING IT. The `if r.get("masks") ...`
block is located with `ast` (by structure, not by a line number), its source
is extracted verbatim, dedented and exec'd in a scope holding nothing but
numpy and planted values. So this gate runs the SHIPPED bytes; it cannot
drift from a copy.

Cases: right-edge overflow, bottom-edge overflow, an interior window (must
stay bit-identical to the legacy paste), a window fully outside the canvas,
and a NEGATIVE case - the pre-A1.32 source is embedded and must still raise,
which is what proves this gate can detect the bug rather than passing always.
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

TILE = 1008

LEGACY_SRC = textwrap.dedent("""
    if r.get("masks") is not None:
        tile_masks = r["masks"]
        any_tile = tile_masks.any(axis=0)
        th, tw = any_tile.shape
        merged[q_idx]["mask_union"][
            yo:yo + th, xo:xo + tw
        ] |= any_tile
""")


def find_infer_sam(explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"check_infer_paste: {p} not found")
        return p
    for p in CANDIDATES:
        if p.exists():
            return p
    sys.exit("check_infer_paste: infer_sam.py not found beside benchmark/ "
             "- pass --file")


def extract_paste_block(path: Path) -> str:
    """Source of the `if r.get("masks") is not None:` block, dedented.

    Located structurally inside predict_sliding_window, so renumbering the
    file cannot silently point this gate at the wrong code.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "predict_sliding_window":
            fn = node
            break
    if fn is None:
        sys.exit(f"check_infer_paste: predict_sliding_window not found in {path}")
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        seg = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if "mask_union" in seg and "masks" in seg:
            return textwrap.dedent(seg)
    sys.exit("check_infer_paste: the mask_union paste block was not found "
             "inside predict_sliding_window - was it renamed or removed?")


def run_paste(block_src: str, canvas_h: int, canvas_w: int,
              yo: int, xo: int, tile: int = TILE):
    """Exec the extracted block against a planted canvas; return the canvas."""
    scope = {
        "np": np,
        "r": {"masks": np.ones((1, tile, tile), dtype=bool)},
        "merged": {0: {"mask_union": np.zeros((canvas_h, canvas_w), dtype=bool)}},
        "q_idx": 0,
        "yo": yo, "xo": xo,
        "H": canvas_h, "W": canvas_w,
    }
    exec(compile(block_src, "<paste-block>", "exec"), scope)
    return scope["merged"][0]["mask_union"]


def selftest(path: Path) -> int:
    block = extract_paste_block(path)
    fails = []

    # 1. right edge: image narrower than a tile - the shape that crashed
    #    (destination (1008, 951) against a (1008, 1008) tile)
    try:
        c = run_paste(block, 1500, 951, yo=0, xo=0)
        if not (c[:TILE, :951].all() and not c[TILE:].any()):
            fails.append("right-edge: wrong region marked")
        else:
            print("  [ok] right edge   canvas 1500x951, window at (0,0)")
    except Exception as e:                                   # noqa: BLE001
        fails.append(f"right-edge raised {type(e).__name__}: {e}")

    # 2. bottom edge: image shorter than a tile
    try:
        c = run_paste(block, 959, 1500, yo=0, xo=0)
        if not (c[:959, :TILE].all() and not c[:, TILE:].any()):
            fails.append("bottom-edge: wrong region marked")
        else:
            print("  [ok] bottom edge  canvas 959x1500, window at (0,0)")
    except Exception as e:                                   # noqa: BLE001
        fails.append(f"bottom-edge raised {type(e).__name__}: {e}")

    # 3. interior window - MUST stay bit-identical to the legacy paste, which
    #    is the guarantee that no already-successful mask moved a pixel
    try:
        new = run_paste(block, 2000, 2000, yo=756, xo=756)
        old = run_paste(LEGACY_SRC, 2000, 2000, yo=756, xo=756)
        if not np.array_equal(new, old):
            fails.append("interior window differs from the legacy paste")
        else:
            print("  [ok] interior     bit-identical to the pre-A1.32 paste")
    except Exception as e:                                   # noqa: BLE001
        fails.append(f"interior raised {type(e).__name__}: {e}")

    # 4. window entirely outside the canvas - must write nothing, not raise
    try:
        c = run_paste(block, 500, 500, yo=500, xo=0)
        if c.any():
            fails.append("outside-canvas: wrote pixels it should not have")
        else:
            print("  [ok] outside      window past the canvas writes nothing")
    except Exception as e:                                   # noqa: BLE001
        fails.append(f"outside-canvas raised {type(e).__name__}: {e}")

    # 5. NEGATIVE case - the pre-A1.32 source must still fail case 1, or this
    #    gate cannot detect the bug it exists for and a PASS means nothing
    try:
        run_paste(LEGACY_SRC, 1500, 951, yo=0, xo=0)
        fails.append("NEGATIVE case did not raise - the harness cannot "
                     "detect the bug, so a PASS here would mean nothing")
    except ValueError:
        print("  [ok] negative     pre-A1.32 source still raises ValueError")
    except Exception as e:                                   # noqa: BLE001
        fails.append(f"negative case raised {type(e).__name__}, wanted "
                     f"ValueError: {e}")

    if fails:
        print("\ncheck_infer_paste: FAIL")
        for f in fails:
            print("  - " + f)
        print("\nThis is a CODE fault in predict_sliding_window's mask paste, "
              "not an environment or data problem. Fix it in the repo and "
              "git pull - never patch the rented box (collect_results pins "
              "the git SHA as provenance).")
        return 1
    print(f"check_infer_paste: PASS ({path.name})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="path to infer_sam.py (default: beside benchmark/)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest(find_infer_sam(a.file))


if __name__ == "__main__":
    raise SystemExit(main())
