# -*- coding: utf-8 -*-
"""check_lora_effect - prove the LoRA weights actually reach the model
(Amendment A1.24 item 136).

Two mask dirs produced by run_a5_zeroshot on the SAME images:
  --a6  run with --weights <checkpoint>   (LoRA applied and loaded)
  --a5  run with no --weights             (--no-lora, base SAM3)

If those masks come out pixel-identical, one of the two silent failures of
A1.24 has happened:
  * the checkpoint matched nothing and load_state_dict(strict=False) swallowed
    it (item 132.4), or
  * the run labelled "A6" never had any weights at all (item 132.1/132.3).
Either way row A6 would be row A5, and a5_vs_a6.csv - the whole point of the
interim rental - would report "LoRA adds nothing" with no error anywhere.

Outcomes:
  DIFFER            pass - the two configurations produce different masks
  IDENTICAL         fail - non-empty but identical: the weights did not land
  BOTH EMPTY        fail - inconclusive; nothing was detected either way, so
                    the comparison proves nothing (do not run the queue on
                    an inconclusive result - check the detection threshold
                    and the smoke images first)

Usage:
  python check_lora_effect.py --a6 /workspace/smoke/m_a6 \
                              --a5 /workspace/smoke/m_a5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def masks(d: Path) -> dict:
    return {p.name[:-len("_mask.png")]: p
            for p in sorted(d.glob("*_mask.png"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a6", type=Path, required=True)
    ap.add_argument("--a5", type=Path, required=True)
    a = ap.parse_args()

    m6, m5 = masks(a.a6), masks(a.a5)
    common = sorted(set(m6) & set(m5))
    if not common:
        print(f"FAIL: no image has a mask in BOTH dirs "
              f"(a6={len(m6)} a5={len(m5)}) - inference did not write masks")
        return 1

    n_diff = n_same = n_empty = 0
    for stem in common:
        x = cv2.imread(str(m6[stem]), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(str(m5[stem]), cv2.IMREAD_GRAYSCALE)
        if x is None or y is None:
            print(f"FAIL: unreadable mask for {stem}")
            return 1
        if x.shape != y.shape:
            print(f"{stem}: shapes differ {x.shape} vs {y.shape} -> DIFFER")
            n_diff += 1
            continue
        px6, px5 = int((x > 0).sum()), int((y > 0).sum())
        if np.array_equal(x, y):
            if px6 == 0:
                n_empty += 1
                print(f"{stem}: both masks EMPTY (inconclusive)")
            else:
                n_same += 1
                print(f"{stem}: IDENTICAL, {px6} px each")
        else:
            n_diff += 1
            inter = int(((x > 0) & (y > 0)).sum())
            union = int(((x > 0) | (y > 0)).sum())
            print(f"{stem}: differ - a6 {px6} px, a5 {px5} px, "
                  f"IoU {inter / union:.3f}" if union else f"{stem}: differ")

    print(f"\n{len(common)} shared image(s): {n_diff} differ, "
          f"{n_same} identical, {n_empty} both-empty")
    if n_diff:
        print("check_lora_effect PASS: the LoRA checkpoint changes the output, "
              "so the weights reached the model and row A6 is not row A5")
        return 0
    if n_empty == len(common):
        print("check_lora_effect FAIL (INCONCLUSIVE): every mask is empty in "
              "both modes - nothing was detected, so this proves nothing. "
              "Do NOT read it as 'LoRA has no effect'.")
        return 1
    print("check_lora_effect FAIL: identical non-empty masks - the weights did "
          "NOT reach the model (A1.24 item 132.3/132.4). Row A6 would be row "
          "A5. Stop; do not release the queue.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
