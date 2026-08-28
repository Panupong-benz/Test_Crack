# -*- coding: utf-8 -*-
"""Axis C driver — the benchmark's task metric ("selling point", protocol
SS4): each model's masks -> production geometry chain -> x1 -> frozen
sqrt-law -> |Rp_pred - Rmea| per step, against the field-tier yardstick
(SAM3-LoRA reproduces the field tier at MAE 0.227, 1.00x — CLAUDE.md 8u).

RW20 only (Amendment A1.2: the one wall with canvas naming). Frozen
k = 2.7902 — never refit. Runs LOCALLY (mechanics chain + canvas naming
are not on the vast.ai instance).

Pipeline replayed with the STORED production transforms — nothing is
re-detected, so the geometry is identical to the SS8 chain by construction:
  full-frame predicted mask
    -> lens undistort with the image's stored coeffs (<name>_coeffs.json)
    -> the stored out_crop rectangle
    -> cv2.warpPerspective with the stored _H.npz homography (NEAREST)
    -> parallel tree data/out_crop_bm/<tag>/rectified/
    -> fuse_canvas --rect ... --out (parallel fused/)
    -> link_canvas --fused ...
    -> run_mechanics_chain --fused ... --out ...
    -> Z1..Z4_Wr -> x1 (rb_from_zones, same form as audit_vision_identity)
    -> Rp_pred = 2.7902 * sign(x1) * sqrt(|x1|)
    -> results/benchmark/axis_c_<tag>.csv

Gates (both runnable TODAY, before any benchmark mask exists):
  --gate front   production full-frame masks (preds_RW20_fullframe) through
                 the front end must reproduce the existing _rect_mask.png
                 files pixel-exactly.
  --gate chain   the existing production _rect_mask.png set through
                 fuse -> link -> chain must reproduce the field x1 to
                 <= 5e-4 on every step and land at the 0.227 yardstick.
A failed gate means STOP AND REPORT — never force through.

Usage:
  python benchmark/axis_c_rp_error.py --gate front
  python benchmark/axis_c_rp_error.py --gate chain
  python benchmark/axis_c_rp_error.py --pred runs/a6_RW20_s0/masks --tag a6_RW20_s0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent          # benchmark/
ROOT = HERE.parent                              # THESIS_crack_tool/
CODE = ROOT / "code"
CORRECTED = ROOT / "data" / "out_crop" / "corrected"
RECT = ROOT / "data" / "out_crop" / "rectified"
BM_ROOT = ROOT / "data" / "out_crop_bm"
RESULTS = ROOT / "results" / "benchmark"
SR_DATASET = ROOT / "results" / "SR" / "sr_dataset.csv"

K_FROZEN = 2.7902          # SS8i pooled PySR constant — NEVER refit here
YARDSTICK_MAE = 0.227      # field tier / SS8u vision identity
TOL_X1 = 5e-4

sys.path.insert(0, str(CODE))


def core_of(name: str):
    m = re.match(r"(IMG_\d+)", name)
    return m.group(1) if m else None


def rb_from_zones(zw, xn=297.0, lw=900.0, hw=1800.0, hz=450.0):
    """Route-B drift from zone Wr sums — same form as audit_vision_identity."""
    lever = [hw - (i + 0.5) * hz for i in range(4)]
    phi = [w / hz / (lw - xn) for w in zw]
    return sum(p * hz * lv for p, lv in zip(phi, lever)) / hw * 100.0


# ------------------------------------------------------------ front end ---
def stored_params(core: str):
    """coeffs.json (undistort params + crop rect) and _H.npz + canvas size
    for one image core; None if the image never made it through Stage A."""
    cj = glob.glob(str(CORRECTED / f"{core}_*coeffs.json"))
    hp = RECT / f"{core}_H.npz"
    rp = RECT / f"{core}_rect.png"
    if not cj or not hp.exists() or not rp.exists():
        return None
    meta = json.loads(Path(cj[0]).read_text(encoding="utf-8"))
    d = np.load(hp, allow_pickle=True)
    rect_shape = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).shape
    return {"xc": meta["xcenter"], "yc": meta["ycenter"],
            "coeffs": meta["coeffs_backward"],
            "crop": meta.get("out_crop"), "H": d["H"],
            "canvas_wh": (rect_shape[1], rect_shape[0])}


def mask_to_rect(mask: np.ndarray, p: dict) -> np.ndarray:
    """Full-frame mask -> rectified raster, replaying the stored transforms
    exactly as write_corrected + auto_rectify did (float undistort, >127
    binarize, stored crop rect, NEAREST perspective warp)."""
    import discorpy.post.postprocessing as post
    um = post.unwarp_image_backward(mask.astype(np.float32),
                                    p["xc"], p["yc"], p["coeffs"])
    um = (np.clip(um, 0, 255) > 127).astype(np.uint8) * 255
    if p["crop"]:
        cx, cy, cw, ch = p["crop"]
        um = um[cy:cy + ch, cx:cx + cw]
    return cv2.warpPerspective(um, np.asarray(p["H"], float), p["canvas_wh"],
                               flags=cv2.INTER_NEAREST)


def build_parallel_tree(pred_dir: Path, tag: str):
    """Predicted masks -> parallel rectified tree. Returns (tree, coverage)."""
    tree = BM_ROOT / tag / "rectified"
    tree.mkdir(parents=True, exist_ok=True)
    n_in = n_ok = 0
    for f in sorted(pred_dir.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        core = core_of(f.name)
        if core is None:
            continue
        n_in += 1
        p = stored_params(core)
        if p is None:
            continue                       # image never rectified in Stage A
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        rect = mask_to_rect(mask, p)
        cv2.imwrite(str(tree / f"{core}_rect_mask.png"), rect)
        for side in (f"{core}_H.npz", f"{core}_rectify.json"):
            src = RECT / side
            if src.exists():
                shutil.copy2(src, tree / side)
        n_ok += 1
    print(f"[front] {n_ok}/{n_in} masks entered the wall frame "
          f"(rest: no Stage-A rectification for that image)")
    return tree, (n_ok, n_in)


# ------------------------------------------------------------ chain part ---
def run_chain(tree: Path, tag: str) -> Path:
    fused = tree.parent / "fused"
    mech_csv = tree.parent / "mechanics_chain.csv"
    steps = [
        [sys.executable, str(CODE / "fuse_canvas.py"),
         "--rect", str(tree), "--out", str(fused)],
        [sys.executable, str(CODE / "link_canvas.py"), "--fused", str(fused)],
        [sys.executable, str(CODE / "run_mechanics_chain.py"),
         "--fused", str(fused), "--out", str(mech_csv)],
    ]
    for cmd in steps:
        print("+ " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            raise RuntimeError(f"chain step failed: {cmd[1]}")
    return mech_csv


def field_k():
    """The SS8u yardstick's constant: LS k pooled on the FIELD tier
    (vision_case_study.py step 1 — never fitted on model masks)."""
    num = den = 0.0
    with open(SR_DATASET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["abs_Rp2_ok"] != "1"
                    or r["RB_r_keep_Xn297_pct"] in ("", "nan")):
                continue
            x = float(r["RB_r_keep_Xn297_pct"])
            y = float(r["Rp2_mea_pct"])
            s = math.copysign(math.sqrt(abs(x)), x)
            num += y * s
            den += s * s
    return num / den


def score(mech_csv: Path, tag: str, out_csv: Path):
    """mechanics CSV -> x1 -> Rp errors under BOTH constants:
    - Rp_pred (K_FROZEN = 2.7902): the protocol's primary, never refit;
    - Rp_pred_fieldk (LS k on the field tier): the SS8u yardstick's own
      convention, so the 0.227 comparison stays apples-to-apples.
    Scoring basis = vision_case_study's (every step with |Rmea| >= 0.25);
    x1 == 0 rows additionally carry status NO_RESIDUAL_SIGNAL (SS8aq)."""
    kf = field_k()
    field, rp_meas = {}, {}
    with open(SR_DATASET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["wall"] != "RW20":
                continue
            d = float(r["drift"])                 # already signed in sr_dataset
            if r.get("RB_r_keep_Xn297_pct") not in ("", "nan", None):
                field[d] = float(r["RB_r_keep_Xn297_pct"])
            if r.get("Rp2_mea_pct") not in ("", "nan", None):
                rp_meas[d] = float(r["Rp2_mea_pct"])

    def law(x, k):
        return k * math.copysign(math.sqrt(abs(x)), x)

    rows = []
    with open(mech_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = float(r["drift"]) * (1 if r["sign"] == "+" else -1)
            zw = [float(r[f"Z{i}_Wr"]) for i in range(1, 5)]
            x1 = math.copysign(rb_from_zones(zw), d)
            status = "NO_RESIDUAL_SIGNAL" if x1 == 0 else "ok"
            scored = d in rp_meas and abs(rp_meas[d]) >= 0.25
            row = {"tag": tag, "step": f"{r['drift']}{r['sign']}",
                   "drift": d, "x1": round(x1, 6),
                   "x1_field": field.get(d, ""), "status": status,
                   "Rp_pred": round(law(x1, K_FROZEN), 4),
                   "Rp_pred_fieldk": round(law(x1, kf), 4),
                   "Rmea": rp_meas.get(d, "")}
            row["abs_err"] = (round(abs(row["Rp_pred"] - rp_meas[d]), 4)
                              if scored else "")
            row["abs_err_fieldk"] = (
                round(abs(row["Rp_pred_fieldk"] - rp_meas[d]), 4)
                if scored else "")
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "step", "drift", "x1",
                                          "x1_field", "status", "Rp_pred",
                                          "Rp_pred_fieldk", "Rmea",
                                          "abs_err", "abs_err_fieldk"])
        w.writeheader()
        w.writerows(rows)
    e_fk = [r["abs_err_fieldk"] for r in rows if r["abs_err_fieldk"] != ""]
    e_fz = [r["abs_err"] for r in rows if r["abs_err"] != ""]
    mae_fk = sum(e_fk) / len(e_fk) if e_fk else float("nan")
    mae_fz = sum(e_fz) / len(e_fz) if e_fz else float("nan")
    n_nosig = sum(1 for r in rows if r["status"] == "NO_RESIDUAL_SIGNAL")
    print(f"[axis C] {tag}: n={len(e_fk)} scored, MAE fieldk {mae_fk:.4f} "
          f"(yardstick {YARDSTICK_MAE}) | frozen-k {mae_fz:.4f} | "
          f"no-signal steps {n_nosig} -> {out_csv}")
    return rows, mae_fk


# --------------------------------------------------------------- gates ---
def gate_front():
    """Production full-frame masks -> front end must equal the existing
    _rect_mask.png files pixel-exactly."""
    src = ROOT / "data" / "preds_RW20_fullframe"
    if not src.exists():
        print(f"GATE front: {src} missing — cannot run")
        return 1
    bad = ok = 0
    for rm in sorted(RECT.glob("*_rect_mask.png")):
        core = core_of(rm.name)
        cand = (list(src.glob(f"{core}*_mask.png"))
                or list(src.glob(f"{core}*.png")))
        if not cand:
            continue
        p = stored_params(core)
        if p is None:
            continue
        mask = cv2.imread(str(cand[0]), cv2.IMREAD_GRAYSCALE)
        got = mask_to_rect(mask, p)
        ref = cv2.imread(str(rm), cv2.IMREAD_GRAYSCALE)
        diff = int(((got > 127) != (ref > 127)).sum())
        if diff:
            bad += 1
            print(f"  MISMATCH {core}: {diff} px differ")
        else:
            ok += 1
    print(f"GATE front: {ok} exact, {bad} mismatched")
    if bad or ok == 0:
        print("GATE front FAILED — stop and report, do not force")
        return 1
    print("GATE front PASS — front end replays production exactly")
    return 0


def gate_chain():
    """Existing production rect masks -> fuse/link/chain must reproduce the
    field x1 (<=5e-4) and the 0.227 yardstick."""
    tag = "_gate_chain"
    tree = BM_ROOT / tag / "rectified"
    if tree.exists():
        shutil.rmtree(tree.parent)
    tree.mkdir(parents=True)
    n = 0
    for rm in sorted(RECT.glob("*_rect_mask.png")):
        core = core_of(rm.name)
        for side in (rm.name, f"{core}_H.npz", f"{core}_rectify.json"):
            src = RECT / side
            if src.exists():
                shutil.copy2(src, tree / side)
        n += 1
    print(f"GATE chain: {n} production rect masks copied")
    mech = run_chain(tree, tag)
    rows, mae = score(mech, tag, RESULTS / f"axis_c{tag}.csv")
    bad = [r for r in rows if r["x1_field"] != "" and r["x1"] != ""
           and abs(float(r["x1"]) - float(r["x1_field"])) > TOL_X1]
    for r in bad:
        print(f"  x1 mismatch {r['step']}: {r['x1']} vs field "
              f"{r['x1_field']}")
    if bad or not (abs(mae - YARDSTICK_MAE) < 0.001):
        print(f"GATE chain FAILED (mismatches={len(bad)}, MAE {mae:.4f} vs "
              f"{YARDSTICK_MAE}) — stop and report, do not force")
        return 1
    print(f"GATE chain PASS — x1 identity holds on every step, "
          f"MAE {mae:.4f} = yardstick")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["front", "chain"], default=None)
    ap.add_argument("--pred", type=Path, help="model's full-frame masks")
    ap.add_argument("--tag", help="e.g. a6_RW20_s0")
    args = ap.parse_args()
    if args.gate == "front":
        return gate_front()
    if args.gate == "chain":
        return gate_chain()
    if not (args.pred and args.tag):
        print("need --pred and --tag (or --gate front|chain)")
        return 2
    tree, _cov = build_parallel_tree(args.pred, args.tag)
    mech = run_chain(tree, args.tag)
    score(mech, args.tag, RESULTS / f"axis_c_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
