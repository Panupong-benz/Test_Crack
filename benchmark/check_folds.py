# -*- coding: utf-8 -*-
"""Gate: the folds rebuilt on the GPU instance must reproduce the ones the
pool was frozen against (benchmark_protocol.md Amendment A1.3).

We upload the POOL, not the folds - each LOWO fold holds the whole pool, so
shipping four of them uploads every photo four times. lowo_split is
deterministic (SEED=42, GROUP_BY_STEP), so the instance rebuilds them with
symlinks and this compares the result against folds_summary_expected.json,
which pack_pool.py put in the archive. That proves the SPLIT LOGIC
reproduced, which a zip checksum cannot.

  python check_folds.py --expected folds_summary_expected.json \
                        --got folds/folds_summary.json
  python check_folds.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ("train", "valid", "test", "leak_images", "straddle_loadsteps")


def compare(exp: dict, got: dict):
    bad = []
    for fold, e in exp.items():
        g = got.get(fold)
        if g is None:
            bad.append(f"{fold}: fold missing on this machine")
            continue
        for k in KEYS:
            if e.get(k) != g.get(k):
                bad.append(f"{fold}.{k}: expected {e.get(k)}, got {g.get(k)}")
        if g.get("leak_images") or g.get("straddle_loadsteps"):
            bad.append(f"{fold}: leak={g.get('leak_images')} "
                       f"straddle={g.get('straddle_loadsteps')} (must be 0)")
    for fold in got:
        if fold not in exp:
            bad.append(f"{fold}: unexpected extra fold")
    return bad


def selftest():
    exp = {"RW20": {"train": 161, "valid": 37, "test": 108,
                    "leak_images": 0, "straddle_loadsteps": 0}}
    assert compare(exp, exp) == []
    bad = compare(exp, {"RW20": {**exp["RW20"], "valid": 36}})
    assert len(bad) == 1 and "valid" in bad[0], bad
    bad = compare(exp, {"RW20": {**exp["RW20"], "leak_images": 1}})
    assert any("leak" in b for b in bad), bad
    assert compare(exp, {}) and compare({}, exp)
    print("selftest PASS: identical passes; count drift, leakage, missing "
          "and extra folds all fail")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", type=Path)
    ap.add_argument("--got", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.expected and a.got):
        ap.error("--expected and --got are required")
    exp = json.loads(a.expected.read_text(encoding="utf-8"))
    got = json.loads(a.got.read_text(encoding="utf-8"))
    bad = compare(exp, got)
    print(f"fold gate: {'FAIL' if bad else 'PASS'} - "
          f"{len(exp)} folds rebuilt from the pool")
    for b in bad:
        print(f"   {b}")
    if not bad:
        for fold, e in sorted(exp.items()):
            print(f"   {fold}: {e['train']}/{e['valid']}/{e['test']} "
                  f"(train/valid/test), 0 leak, 0 straddle")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
