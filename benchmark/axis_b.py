# -*- coding: utf-8 -*-
"""Axis B — external transfer, both directions. Inference only, no
training, no fitting (benchmark_protocol.md SS5 + Amendment A1.2).

B1  our SAM3-LoRA checkpoint -> external crack sets (OmniCrack30k test,
    Road420, Facade390, Concrete3k). Prediction P-B1: clearly worse than
    their trained nnU-Net — we learned PEN TRACE, not crack texture.
B2  the released OmniCrack30k nnU-Net -> our 4 fold-wall test sets +
    marked-line subset. Prediction P-B2: heavy FP on grid lines / written
    numbers; marked-FP% worse than every domain-trained A row.
Both predictions are reported straight whether right or wrong.

Data acquisition is MANUAL on the instance (URLs in vastai_runbook.md):
external images+GT-mask dirs, and the released nnU-Net model folder from
github.com/ben-z-original/omnicrack30k. This script only runs inference
and pipes everything through the one evaluator.

Usage:
  python axis_b.py b1 --weights <best_lora_weights.pt> \
      --images <ext_images_dir> --gt-mask-dir <ext_gt_dir> \
      --dataset omnicrack30k [--threshold 0.5] [--results results/benchmark]
  python axis_b.py b2 --nnunet-model <released_model_dir> \
      --fold data/fold_RW20 [--marked-list marked_line_images.txt] \
      [--results results/benchmark]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"}


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd])


def cmd_b1(args):
    """Our checkpoint over an external set, via the same infer_sam shim."""
    out_masks = Path(f"runs/b1_{args.dataset}/masks")
    imgs = sorted(p for p in args.images.iterdir() if p.suffix in EXTS)
    out_masks.mkdir(parents=True, exist_ok=True)
    infer = None
    for c in (HERE.parent / "code" / "infer_sam.py",
              HERE.parent / "infer_sam.py"):
        if c.exists():
            infer = c
            break
    failures = []
    for i, img in enumerate(imgs):
        cmd = [sys.executable, infer, "--image", img, "--prompt", "crack",
               "--output", out_masks / img.stem,
               "--threshold", args.threshold, "--sliding-window",
               "--tile-size", 1008, "--tile-overlap", 0.25,
               # without --save-mask infer_sam writes only the overlay figure
               # and eval_masks would score THAT (Amendment A1.4)
               "--save-mask",
               "--no-progress", "--weights", args.weights]
        r = run(cmd)
        print(f"[{i + 1}/{len(imgs)}] {img.name}: "
              f"{'ok' if r.returncode == 0 else 'FAIL'}")
        if r.returncode != 0:
            failures.append(img.name)
    (out_masks / "b1_run.json").write_text(json.dumps(
        {"dataset": args.dataset, "n": len(imgs),
         "failures": failures}, indent=2))
    ev = run([sys.executable, HERE / "eval_masks.py",
              "--gt-mask-dir", args.gt_mask_dir, "--pred", out_masks,
              "--out", Path(args.results) / f"eval_b1_{args.dataset}.csv"])
    return 1 if (failures or ev.returncode) else 0


def cmd_b2(args):
    """Released OmniCrack30k nnU-Net over our fold test (+ marked subset).
    Uses nnUNetv2_predict with an explicit model folder (-m), never a
    locally trained dataset id."""
    fold_wall = args.fold.name.replace("fold_", "")
    out_masks = Path(f"runs/b2_{fold_wall}/masks")
    out_masks.mkdir(parents=True, exist_ok=True)
    # nnUNet expects <case>_0000.<ext> input naming; stage via to_nnunet's
    # imagesTs convention if not already done
    stage = Path(f"runs/b2_{fold_wall}/imagesTs")
    if not stage.exists():
        stage.mkdir(parents=True)
        import cv2
        for p in sorted((args.fold / "test").iterdir()):
            if p.suffix in EXTS:
                img = cv2.imread(str(p))
                if img is not None:
                    cv2.imwrite(str(stage / f"{p.stem}_0000.png"), img)
    r = run(["nnUNetv2_predict", "-i", stage, "-o", out_masks,
             "-m", args.nnunet_model, "-f", "all"])
    if r.returncode != 0:
        print("nnUNetv2_predict failed — check the released model folder "
              "layout (-m flag semantics differ across nnUNet versions; "
              "adjust here once, everything still flows through eval)")
        return 1
    ev = run([sys.executable, HERE / "eval_masks.py",
              "--gt", args.fold / "test", "--pred", out_masks,
              "--out", Path(args.results) / f"eval_b2_{fold_wall}.csv"]
             + (["--marked-list", args.marked_list]
                if args.marked_list else []))
    return ev.returncode


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b1 = sub.add_parser("b1")
    b1.add_argument("--weights", type=Path, required=True)
    b1.add_argument("--images", type=Path, required=True)
    b1.add_argument("--gt-mask-dir", type=Path, required=True)
    b1.add_argument("--dataset", required=True,
                    help="omnicrack30k | road420 | facade390 | concrete3k")
    b1.add_argument("--threshold", type=float, default=0.5)
    b1.add_argument("--results", default="results/benchmark")

    b2 = sub.add_parser("b2")
    b2.add_argument("--nnunet-model", type=Path, required=True,
                    help="released OmniCrack30k nnU-Net model folder")
    b2.add_argument("--fold", type=Path, required=True)
    b2.add_argument("--marked-list", default=None)
    b2.add_argument("--results", default="results/benchmark")

    args = ap.parse_args()
    return cmd_b1(args) if args.cmd == "b1" else cmd_b2(args)


if __name__ == "__main__":
    raise SystemExit(main())
