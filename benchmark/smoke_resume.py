# -*- coding: utf-8 -*-
"""smoke_resume - prove exact resume on the REAL trainer + GPU before the queue
(Amendment A1.22 item 120).

Runs after `smoke_sam3` (smoke_test.sh: 1 epoch on the tiny smoke fold, writing
outputs/sam3_lora_full/ckpt_state.pt). Three steps:

  1. resume:   same smoke config with num_epochs=2 and --resume
               -> must continue from epoch 1, train exactly one more epoch,
                  val_stats.json = epochs [1, 2], run.json.resumed_from == [1]
  2. straight: a fresh 2-epoch run into a SEPARATE output dir
  3. compare:  epoch-2 data fingerprints must be IDENTICAL (exact, from the raw
               image batch) - the resumed epoch saw exactly what the
               uninterrupted run saw. val_loss is printed but only loosely
               compared (cudnn.benchmark => not bitwise).

This is the AdamW8bit state round-trip that cannot be tested on the dev box
(no bitsandbytes there). Exit 1 on any mismatch; the queue's kill-gate then
stops before real money is spent.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _records(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _run(cmd, label):
    print(f"[smoke_resume] {label}: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[smoke_resume] {label} exited {rc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke-config", default="/workspace/smoke/smoke_config.yaml")
    ap.add_argument("--trainer", default="train_sam3_lora_native_claude.py")
    ap.add_argument("--val-tol", type=float, default=0.10,
                    help="loose relative tolerance on epoch-2 val_loss (cudnn)")
    a = ap.parse_args()

    base = yaml.safe_load(Path(a.smoke_config).read_text(encoding="utf-8"))
    out_dir = Path(base["output"]["output_dir"])
    state = out_dir / "ckpt_state.pt"
    if not state.exists():
        raise SystemExit(f"[smoke_resume] {state} missing - smoke_sam3 must run "
                         f"first (and the trainer must be the A1.22 build)")
    recs = _records(out_dir / "val_stats.json")
    if [r["epoch"] for r in recs] != [1]:
        raise SystemExit(f"[smoke_resume] expected val_stats epochs [1] after "
                         f"smoke_sam3, got {[r['epoch'] for r in recs]}")

    # 1. resume to 2 epochs in the SAME dir
    cfg_r = yaml.safe_load(Path(a.smoke_config).read_text(encoding="utf-8"))
    cfg_r["training"]["num_epochs"] = 2
    p_r = Path(a.smoke_config).with_name("smoke_config_e2.yaml")
    p_r.write_text(yaml.safe_dump(cfg_r, sort_keys=False), encoding="utf-8")
    _run([sys.executable, a.trainer, "--config", str(p_r), "--resume"], "resume 1->2")

    recs = _records(out_dir / "val_stats.json")
    ep = [r["epoch"] for r in recs]
    if ep != [1, 2]:
        raise SystemExit(f"[smoke_resume] FAIL: val_stats epochs {ep}, want [1, 2] "
                         f"(rotation fired, or the run restarted at 0)")
    rj = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    if rj.get("resumed_from") != [1] or rj.get("epochs") != 2:
        raise SystemExit(f"[smoke_resume] FAIL: run.json {rj}")
    fp_resumed = recs[1].get("fingerprint")
    vl_resumed = recs[1]["val_loss"]

    # 2. straight 2-epoch run in a separate dir (fresh seed, same data)
    cfg_s = yaml.safe_load(Path(a.smoke_config).read_text(encoding="utf-8"))
    cfg_s["training"]["num_epochs"] = 2
    cfg_s["output"]["output_dir"] = str(out_dir.with_name(out_dir.name + "_straight2"))
    p_s = Path(a.smoke_config).with_name("smoke_config_straight2.yaml")
    p_s.write_text(yaml.safe_dump(cfg_s, sort_keys=False), encoding="utf-8")
    _run([sys.executable, a.trainer, "--config", str(p_s)], "straight 2 epochs")
    recs_s = _records(Path(cfg_s["output"]["output_dir"]) / "val_stats.json")
    if [r["epoch"] for r in recs_s] != [1, 2]:
        raise SystemExit(f"[smoke_resume] straight run wrote {recs_s}")
    fp_straight = recs_s[1].get("fingerprint")
    vl_straight = recs_s[1]["val_loss"]

    # 3. compare
    print(f"[smoke_resume] epoch-2 fingerprint resumed={fp_resumed} "
          f"straight={fp_straight}")
    print(f"[smoke_resume] epoch-2 val_loss    resumed={vl_resumed:.6f} "
          f"straight={vl_straight:.6f}")
    if fp_resumed is None or fp_straight is None:
        raise SystemExit("[smoke_resume] FAIL: fingerprint missing - is the "
                         "trainer the A1.22 build?")
    if fp_resumed != fp_straight:
        raise SystemExit("[smoke_resume] FAIL: the resumed epoch 2 did NOT see the "
                         "same data/augmentation as a straight epoch 2 - RNG or "
                         "loader state is not being restored exactly")
    rel = abs(vl_resumed - vl_straight) / max(abs(vl_straight), 1e-9)
    if rel > a.val_tol:
        print(f"[smoke_resume] WARN: epoch-2 val_loss differs by {rel:.1%} "
              f"(> {a.val_tol:.0%}); data identity holds, so this is optimizer/"
              f"kernel nondeterminism - investigate before trusting extension")
        return 1
    print("[smoke_resume] PASS: exact resume proven on the real trainer "
          "(identical epoch-2 data fingerprint; AdamW8bit state round-trip OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
