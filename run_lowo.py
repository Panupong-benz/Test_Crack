#!/usr/bin/env python3
"""
run_lowo.py — Drive a full Leave-One-Wall-Out training sweep for SAM3-LoRA.

For every fold produced by lowo_split.py it:
  1. clones a base config, overriding only:
       training.data_dir   -> <lowo_root>/fold_<W>
       output.output_dir   -> <runs_dir>/fold_<W>
       (optionally training.num_epochs via --epochs)
  2. writes that config next to where the weights will live
       <runs_dir>/fold_<W>/config.yaml      (reproducible: exact config used)
  3. runs train_sam3_lora_native_claude.py --config <that config>
  4. tees stdout/stderr to <runs_dir>/fold_<W>/train.log
  5. checks that best_lora_weights.pt landed, and records the result

Everything else in the base config (lora rank, tiling, cldice, sampling,
checkpoint_path, batch_size, ...) is preserved untouched.

Result layout:
  <runs_dir>/
    fold_RW20/   config.yaml  train.log  best_lora_weights.pt  last_lora_weights.pt
    fold_RW20C/  ...
    ...
    lowo_runs_summary.csv

Examples
--------
# dry run: generate the 4 per-fold configs, don't train
python3 run_lowo.py --lowo-root data_lowo --base-config configs/full_lora_config.yaml \
    --runs-dir outputs/lowo --dry-run

# full sweep, single GPU
python3 run_lowo.py --lowo-root data_lowo --base-config configs/full_lora_config.yaml \
    --runs-dir outputs/lowo

# 2-GPU, skip folds already trained, keep going past a failed fold
python3 run_lowo.py --lowo-root data_lowo --base-config configs/full_lora_config.yaml \
    --runs-dir outputs/lowo --device 0 1 --skip-existing --keep-going

# quick smoke test: 2 epochs on one fold
python3 run_lowo.py --lowo-root data_lowo --base-config configs/full_lora_config.yaml \
    --runs-dir outputs/smoke --folds RW20 --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[error] PyYAML is required (the trainer needs it too): "
          "pip install pyyaml")
    sys.exit(1)

BEST_WEIGHT = "best_lora_weights.pt"
ANN_FILE = "_annotations.coco.json"


def discover_folds(lowo_root: Path, only=None):
    """Return sorted list of (wall, fold_dir) for valid fold_* dirs."""
    folds = []
    for d in sorted(lowo_root.glob("fold_*")):
        if not (d / "train" / ANN_FILE).exists():
            print(f"[warn] {d} has no train/{ANN_FILE}, skipping")
            continue
        wall = d.name[len("fold_"):]
        if only and wall not in only:
            continue
        folds.append((wall, d))
    return folds


def make_fold_config(base_cfg: dict, fold_dir: Path, out_dir: Path,
                     epochs=None) -> dict:
    """Deep-ish copy of base config with per-fold overrides."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    cfg["training"]["data_dir"] = str(fold_dir)
    cfg["output"]["output_dir"] = str(out_dir)
    if epochs is not None:
        cfg["training"]["num_epochs"] = int(epochs)
    return cfg


def run_one(python_exe, train_script, cfg_path: Path, device, log_path: Path):
    """Run training, tee output to log_path. Returns exit code + duration."""
    cmd = [python_exe, str(train_script), "--config", str(cfg_path)]
    if device:
        cmd += ["--device"] + [str(d) for d in device]
    print(f"  $ {' '.join(cmd)}")
    print(f"  log -> {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"# {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:           # stream + tee
            sys.stdout.write("    " + line)
            logf.write(line)
        proc.wait()
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Run a full LOWO training sweep.")
    ap.add_argument("--lowo-root", required=True, type=Path,
                    help="Dir containing fold_<wall>/ (output of lowo_split.py).")
    ap.add_argument("--base-config", required=True, type=Path,
                    help="Base YAML config to clone per fold.")
    ap.add_argument("--runs-dir", required=True, type=Path,
                    help="Where per-fold configs/logs/weights go.")
    ap.add_argument("--train-script", type=Path,
                    default=Path("train_sam3_lora_native_claude.py"),
                    help="Trainer to invoke (default: the crack-tuned one).")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable to use.")
    ap.add_argument("--device", nargs="*", default=None,
                    help="GPU ids passed to the trainer, e.g. --device 0 1.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override training.num_epochs for all folds.")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="Only run these walls, e.g. --folds RW20 RW20C.")
    ap.add_argument("--skip-existing", action="store_true",
                    help=f"Skip folds whose {BEST_WEIGHT} already exists.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue to next fold if one fails (default: stop).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate per-fold configs only; do not train.")
    args = ap.parse_args()

    if not args.base_config.exists():
        print(f"[error] base config not found: {args.base_config}")
        sys.exit(2)
    if not args.dry_run and not args.train_script.exists():
        print(f"[error] train script not found: {args.train_script}")
        sys.exit(2)

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    folds = discover_folds(args.lowo_root, set(args.folds) if args.folds else None)
    if not folds:
        print(f"[error] no folds found under {args.lowo_root}")
        sys.exit(2)

    print(f"Base config : {args.base_config}")
    print(f"Trainer     : {args.train_script}")
    print(f"Runs dir    : {args.runs_dir}")
    print(f"Device      : {args.device or 'default (GPU 0)'}")
    print(f"Folds       : {[w for w, _ in folds]}")
    if args.epochs is not None:
        print(f"Epochs ovr  : {args.epochs}")
    print()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for wall, fold_dir in folds:
        out_dir = args.runs_dir / f"fold_{wall}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.yaml"
        best_path = out_dir / BEST_WEIGHT

        cfg = make_fold_config(base_cfg, fold_dir, out_dir, args.epochs)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"=== fold {wall} ===")
        print(f"  data_dir   : {fold_dir}")
        print(f"  output_dir : {out_dir}")
        print(f"  config     : {cfg_path}")

        if args.skip_existing and best_path.exists():
            print(f"  [skip] {BEST_WEIGHT} already exists\n")
            rows.append({"fold": wall, "status": "skipped",
                         "seconds": 0, "weights": str(best_path)})
            continue

        if args.dry_run:
            print("  [dry-run] config written, not training\n")
            rows.append({"fold": wall, "status": "dry-run",
                         "seconds": 0, "weights": ""})
            continue

        rc, secs = run_one(args.python, args.train_script, cfg_path,
                           args.device, out_dir / "train.log")
        ok = (rc == 0) and best_path.exists()
        status = "ok" if ok else f"FAILED(rc={rc}" + \
                 ("" if best_path.exists() else ",no-weights") + ")"
        print(f"  -> {status}  ({secs/60:.1f} min)\n")
        rows.append({"fold": wall, "status": status, "seconds": round(secs, 1),
                     "weights": str(best_path) if best_path.exists() else ""})

        if not ok and not args.keep_going:
            print(f"[stop] fold {wall} failed and --keep-going not set.")
            break

    # summary
    summ = args.runs_dir / "lowo_runs_summary.csv"
    with open(summ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "status", "seconds", "weights"])
        w.writeheader()
        w.writerows(rows)

    print("=" * 60)
    print("LOWO sweep summary")
    for r in rows:
        print(f"  {r['fold']:8s} {r['status']:24s} {r['weights']}")
    print(f"\nSummary CSV: {summ}")
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    if not args.dry_run:
        print(f"Trained OK: {n_ok}/{len(rows)}")
        print("\nNext: validate each fold on its held-out test set, e.g.")
        print("  python3 validate_sam3_lora.py --config "
              f"{args.runs_dir}/fold_<W>/config.yaml \\")
        print(f"    --weights {args.runs_dir}/fold_<W>/{BEST_WEIGHT} \\")
        print(f"    --val_data_dir {args.lowo_root}/fold_<W>/test")


if __name__ == "__main__":
    main()
