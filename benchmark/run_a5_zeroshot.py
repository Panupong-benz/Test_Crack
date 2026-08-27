# -*- coding: utf-8 -*-
"""Row A5: SAM3 base — NO LoRA — over a fold's test images (isolates what
LoRA + domain training add; the A5/A6 pair is the benchmark's key ablation).

Thin shim over the canonical production inference (code/infer_sam.py,
CLAUDE.md SS4: the 57 KB copy is canonical) so A5 uses EXACTLY the production
prompt / sliding-window / threshold path, minus the LoRA weights.

Usage:
  python run_a5_zeroshot.py --fold <fold_dir> --out <masks_dir> \
      [--prompt crack] [--threshold 0.5] [--extra "--tta ..."]
Exact flag semantics (incl. how infer_sam behaves with no --weights) are
confirmed at smoke hour; if base-model loading needs a flag change, amend
HERE only — every row must keep flowing through infer_sam.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# infer_sam.py: code/ subdir in the local thesis tree, repo root on a
# Test_Crack clone (byte-identical modulo CRLF, verified 2026-08-28)
_ROOT = Path(__file__).resolve().parents[1]
for INFER in (_ROOT / "code" / "infer_sam.py", _ROOT / "infer_sam.py"):
    if INFER.exists():
        break
else:
    raise FileNotFoundError(f"infer_sam.py not found under {_ROOT}")
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default="crack")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    ap.add_argument("--weights", type=Path, default=None,
                    help="LoRA weights -> row A6 inference; omit for A5 "
                         "zero-shot (same shim, one production path)")
    ap.add_argument("--extra", default="", help="extra infer_sam flags")
    args = ap.parse_args()

    test_dir = args.fold / "test"
    imgs = sorted(p for p in test_dir.iterdir() if p.suffix in EXTS)
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, img in enumerate(imgs):
        cmd = [sys.executable, str(INFER),
               "--image", str(img),
               "--prompt", args.prompt,
               "--output", str(args.out / img.stem),
               "--threshold", str(args.threshold),
               "--sliding-window",
               "--tile-size", str(args.tile_size),
               "--tile-overlap", str(args.tile_overlap),
               "--no-progress"]
        if args.weights is not None:
            cmd += ["--weights", str(args.weights)]
        cmd += args.extra.split()
        r = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAIL"
        print(f"[{i + 1}/{len(imgs)}] {img.name}: {status}")
        if r.returncode != 0:
            failures.append({"image": img.name,
                             "tail": (r.stderr or r.stdout)[-400:]})
    (args.out / "a5_run.json").write_text(json.dumps(
        {"n": len(imgs), "failures": failures}, indent=2))
    if failures:
        print(f"{len(failures)} failures — see a5_run.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
