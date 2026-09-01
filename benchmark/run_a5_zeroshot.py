# -*- coding: utf-8 -*-
"""Row A5: SAM3 base — NO LoRA — over a fold's test images (isolates what
LoRA + domain training add; the A5/A6 pair is the benchmark's key ablation).

Thin shim over the canonical production inference (code/infer_sam.py,
CLAUDE.md SS4: the 57 KB copy is canonical) so A5 uses EXACTLY the production
prompt / sliding-window / threshold path, minus the LoRA weights.

Usage:
  python run_a5_zeroshot.py --fold <fold_dir> --out <masks_dir> \
      [--config configs/full_lora_config.yaml] [--weights <lora.pt>] \
      [--prompt crack] [--threshold 0.3] [--limit N] [--extra "--tta ..."]
  python run_a5_zeroshot.py --selftest      # command assembly, no GPU

THE MODES ARE NOT SYMMETRIC (Amendment A1.24 item 133):
  --weights <path>  -> row A6: LoRA applied, that checkpoint loaded
  no --weights      -> row A5: --no-lora is passed to infer_sam, which then
                       applies no LoRA and loads NOTHING. Never leave this to
                       infer_sam's own default: without --no-lora it
                       AUTO-DETECTS outputs/sam3_lora_full/best_lora_weights.pt
                       - the file the smoke hour writes - so "zero-shot" would
                       silently have been a 1-2 epoch smoke checkpoint.

Three things this shim must never get wrong again (Amendments A1.4, A1.24):
  * --config is REQUIRED by infer_sam (required=True). It was never passed,
    so every infer job would have died on argparse - first at job 3 of the
    queue, i.e. after ~22 h of paid training. The default below is the base
    config; make_jobs/a6_adaptive override only data_dir/seed/output_dir/
    num_epochs, so its `lora:` section (rank 16, 12 target modules) matches
    every A6 checkpoint by construction.
  * --save-mask is ALWAYS passed. Without it infer_sam writes only the
    matplotlib overlay figure, which eval_masks would then score as if it
    were the prediction — silent garbage on the most expensive rows.
  * --threshold is infer_sam's DETECTION-CONFIDENCE gate, not a
    mask-probability cut. 0.30 is the production value (thesis report §6
    lowered it from 0.5 for crack recall); 0.5 would quietly handicap
    every SAM3 row.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# infer_sam.py: code/ subdir in the local thesis tree, repo root on a
# Test_Crack clone (byte-identical modulo CRLF, verified 2026-08-28; re-verified 2026-08-31 after A1.16)
_ROOT = Path(__file__).resolve().parents[1]
for INFER in (_ROOT / "code" / "infer_sam.py", _ROOT / "infer_sam.py"):
    if INFER.exists():
        break
else:
    raise FileNotFoundError(f"infer_sam.py not found under {_ROOT}")
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
DEFAULT_CONFIG = _ROOT / "configs" / "full_lora_config.yaml"


def build_cmd(img: Path, args) -> list:
    """The exact argv handed to infer_sam. Extracted so the selftest can
    check it with no model, no GPU and no images (A1.24 item 139)."""
    cmd = [sys.executable, str(INFER),
           "--config", str(args.config),
           "--image", str(img),
           "--prompt", args.prompt,
           # A1.26: the ".png" is load-bearing, not cosmetic. Roboflow names
           # carry dots (IMG_4100_JPG_JPG.rf.<hash>.jpg), so infer_sam's
           # os.path.splitext(args.output) reads the hash as the extension:
           # plt.savefig() then dies with an unsupported format AFTER the
           # whole image has been inferred, and the mask would have landed
           # at "<name>.rf_mask.png" instead of "<stem>_mask.png".
           "--output", str(args.out / (img.stem + ".png")),
           "--threshold", str(args.threshold),
           "--sliding-window",
           "--tile-size", str(args.tile_size),
           "--tile-overlap", str(args.tile_overlap),
           "--save-mask",
           "--no-progress"]
    if args.weights is not None:
        cmd += ["--weights", str(args.weights)]
    else:
        # row A5. NOT infer_sam's default - see the module docstring.
        cmd += ["--no-lora"]
    cmd += args.extra.split()
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fold", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="training config infer_sam builds the model from "
                         "(REQUIRED by infer_sam; default = the base config)")
    ap.add_argument("--prompt", default="crack")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    ap.add_argument("--weights", type=Path, default=None,
                    help="LoRA weights -> row A6 inference; omit for A5 "
                         "zero-shot, which passes --no-lora (same shim, one "
                         "production path)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N test images (smoke probe)")
    ap.add_argument("--extra", default="", help="extra infer_sam flags")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for req in ("fold", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required")
    if not args.config.exists():
        raise SystemExit(f"FATAL: --config {args.config} not found. "
                         f"infer_sam requires it (A1.24 item 132.1).")
    mode = "a6_lora" if args.weights is not None else "a5_zeroshot"
    print(f"[run_a5_zeroshot] mode={mode}  config={args.config}"
          + (f"  weights={args.weights}" if args.weights else "  (--no-lora)"),
          flush=True)

    test_dir = args.fold / "test"
    imgs = sorted(p for p in test_dir.iterdir() if p.suffix in EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    failures, secs = [], []
    t_all = time.time()
    for i, img in enumerate(imgs):
        cmd = build_cmd(img, args)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        secs.append(round(dt, 2))
        status = "ok" if r.returncode == 0 else "FAIL"
        _el = time.time() - t_all
        _eta = (_el / (i + 1)) * (len(imgs) - i - 1)
        print(f"[{i + 1}/{len(imgs)}] {img.name}: {status}  {dt:.1f}s  "
              f"(elapsed {int(_el) // 60}m{int(_el) % 60:02d}s, "
              f"ETA {int(_eta) // 60}m{int(_eta) % 60:02d}s)", flush=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout)[-400:]
            failures.append({"image": img.name, "tail": tail})
            # a5_run.json is not in the queue log, and the queue log is
            # all anyone reads on a rented box
            print(f"       {tail.strip()}", flush=True)
        elif not (args.out / f"{img.stem}_mask.png").exists():
            # exit 0 with no mask means --save-mask silently stopped working;
            # eval would score an empty prediction and never complain
            failures.append({"image": img.name,
                             "tail": "exit 0 but no _mask.png written"})
    # timing here is the ONLY inference-cost number A5/A6 ever produce
    # (predict_seg records ms_per_tile for A2-A4; infer_sam records nothing)
    (args.out / "a5_run.json").write_text(json.dumps(
        {"n": len(imgs), "failures": failures,
         "mode": mode, "config": str(args.config),
         "weights": None if args.weights is None else str(args.weights),
         "threshold": args.threshold,
         "fusion": "or_union+morphology (production infer_sam)",
         "sec_total": round(time.time() - t_all, 1),
         "sec_per_image_mean": (round(sum(secs) / len(secs), 2)
                                if secs else None),
         "sec_per_image": secs}, indent=2))
    if failures:
        print(f"{len(failures)} failures — see a5_run.json")
        sys.exit(1)


def selftest() -> int:
    """Command assembly only - the part that was wrong and that no GPU is
    needed to check."""
    import types
    base = dict(config=Path("configs/full_lora_config.yaml"), prompt="crack",
                out=Path("out"), threshold=0.3, tile_size=1008,
                tile_overlap=0.25, extra="", weights=None)
    img = Path("IMG_1.jpg")

    a5 = build_cmd(img, types.SimpleNamespace(**base))
    a6 = build_cmd(img, types.SimpleNamespace(**{**base,
                                                "weights": Path("best.pt")}))

    for name, cmd in (("A5", a5), ("A6", a6)):
        assert "--config" in cmd, f"{name}: infer_sam requires --config"
        assert cmd[cmd.index("--config") + 1].endswith("full_lora_config.yaml")
        # A1.4: without --save-mask infer_sam writes only its overlay figure
        assert "--save-mask" in cmd, f"{name}: --save-mask missing"
        out = cmd[cmd.index("--output") + 1]
        # A1.26: infer_sam splits the extension off --output for BOTH the
        # overlay figure and the "<base>_mask.png" it writes. A dotted
        # Roboflow name with no extension breaks both.
        assert out.endswith(".png"), f"{name}: --output needs .png, got {out}"
        assert "--no-progress" in cmd and "--sliding-window" in cmd
        assert cmd[cmd.index("--threshold") + 1] == "0.3"

    assert "--no-lora" in a5 and "--weights" not in a5, a5
    assert "--weights" in a6 and "--no-lora" not in a6, a6
    assert a6[a6.index("--weights") + 1] == "best.pt"

    import os as _os
    dotted = Path("IMG_4100_JPG_JPG.rf.dadc3afacbaac24cafbceb75c1b783c6.jpg")
    dcmd = build_cmd(dotted, types.SimpleNamespace(**base))
    dout = dcmd[dcmd.index("--output") + 1]
    dbase, dext = _os.path.splitext(dout)
    assert dext == ".png", dext
    assert Path(dbase + "_mask.png").name == dotted.stem + "_mask.png", dbase

    extra = build_cmd(img, types.SimpleNamespace(**{**base,
                                                   "extra": "--tta --seed 1"}))
    assert extra[-3:] == ["--tta", "--seed", "1"], extra[-3:]

    print("run_a5_zeroshot selftest PASS: --config always passed; A5 asks for "
          "--no-lora and never --weights (so infer_sam cannot auto-detect the "
          "smoke checkpoint); A6 passes --weights and never --no-lora; "
          "--save-mask kept on both; --output always ends .png so a dotted "
          "Roboflow name still yields <stem>_mask.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
