# -*- coding: utf-8 -*-
"""Static guard for the prefix property (Amendment A1.21 item 115).

Every budget/checkpoint claim about row A6 rests on one fact: a run with
num_epochs = B' is THE SAME PROGRAM as the first B' epochs of a run with
num_epochs = 30 - same LR at every step (constant; no scheduler exists),
same data order, same RNG consumption. That is what makes "best_lora_
weights.pt would have been identical" a statement of fact rather than an
approximation, and what lets epoch_saturation's adequacy claim stand.

The hazard is sitting in plain sight: configs/full_lora_config.yaml still
carries `lr_scheduler: cosine` and `warmup_steps` keys, declared DEAD since
A1.4. If anyone ever wires them up, every prefix-property claim already in
the record becomes false RETROACTIVELY. This guard makes that impossible to
do silently.

Checks over train_sam3_lora_native_claude.py:
  1. zero hits for any LR-scheduler construct or per-step LR mutation
  2. `num_epochs` is read exactly once
  3. exactly one epoch loop, `for epoch in range(epochs)`

Exit 1 on any violation, naming the offending lines. Runs in milliseconds,
no torch needed - wired into setup_benchmark.sh's selftest step so it runs
on every rental before money is spent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

def _find_trainer() -> Path:
    """Dual layout (SS8ba pattern): Test_Crack keeps the trainer at repo root;
    the canonical THESIS_crack_tool repo does not hold it - there the trainer
    lives in Result_Coding/22.4.2025 two levels up."""
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "train_sam3_lora_native_claude.py",
                 here.parent.parent.parent / "04_models" / "Result_Coding"
                 / "22.4.2025" / "train_sam3_lora_native_claude.py",
                 here.parent.parent / "Result_Coding" / "22.4.2025"
                 / "train_sam3_lora_native_claude.py"):
        if cand.exists():
            return cand
    raise SystemExit("check_prefix_property: trainer not found in either "
                     "layout - cannot certify the prefix property")


TRAINER = _find_trainer()

FORBID = re.compile(
    r"LambdaLR|CosineAnnealing|OneCycle|ExponentialLR|StepLR|CyclicLR"
    r"|get_cosine|get_linear_schedule|lr_scheduler\.step|\.param_groups")


def main() -> int:
    src = TRAINER.read_text(encoding="utf-8")
    lines = src.splitlines()

    bad = [(i, l.strip()) for i, l in enumerate(lines, 1) if FORBID.search(l)]
    n_read = sum(1 for l in lines if '["num_epochs"]' in l)
    n_loop = sum(1 for l in lines if re.search(r"for epoch in range\(epochs\)", l))

    ok = (not bad) and n_read == 1 and n_loop == 1
    print(f"prefix-property guard over {TRAINER.name}:")
    print(f"  scheduler/LR-mutation hits : {len(bad)} (must be 0)")
    for i, l in bad:
        print(f"    line {i}: {l}")
    print(f"  num_epochs reads           : {n_read} (must be 1)")
    print(f"  'for epoch in range(epochs)': {n_loop} (must be 1)")
    if ok:
        print("PASS: constant LR, single budget read, single epoch loop - "
              "a shorter run is a prefix of a longer one")
        return 0
    print("FAIL: the prefix property no longer holds. Any claim comparing "
          "epoch budgets, or that best_lora_weights.pt at a smaller budget "
          "equals the larger run's early best, is now unsupported "
          "(A1.21 item 115).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
