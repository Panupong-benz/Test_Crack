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

# --wall RW20C runs under pipeline_rerun_spec E3 (which pre-registered
# RW20C/NSW6), NOT under the benchmark's axis C, which Amendment A1.2
# item 18 froze to RW20 - cite E3, never A1.2, for this wall's numbers.
#
# RW20C differences, all verified 2026-09-04 before this was written:
#  * masks are already FULL-frame (the pool holds RW20C uncropped, A1.35)
#    in the EXIF-oriented space Stage A consumed -> --space full, no lift;
#  * only 24 of 52 frames were lens-undistorted (lens column of the batch
#    summary CSV). The rule "undistort iff that frame's coeffs exist" was
#    proven equivalent by md5 of the exact file each _rectify.json names
#    (24 == 24 == 24 across summary column / coeffs set / md5 class), but
#    the SUMMARY column stays authoritative: a deleted coeffs file must be
#    an error for a 15mm frame, never a silent skip;
#  * no out_crop rect (the coeffs schema has none - full-frame warp).
WALLS = {
    "RW20": dict(corrected=CORRECTED, rect=RECT, lens_summary=None,
                 meta=None),
    "RW20C": dict(corrected=ROOT / "data" / "out_rw20c" / "barrel",
                  rect=ROOT / "data" / "out_rw20c" / "rectified",
                  lens_summary=ROOT / "data" / "out_rw20c" /
                  "rw20c_rectification_summary.csv",
                  meta=ROOT / "data" / "out_rw20c" /
                  "meta_stageA_RW20C.csv"),
}


def lens_map(wall_cfg):
    """core -> lens string from the Stage-A batch summary ('' / 'none' =
    the frame ran raw). None when the wall has no summary (RW20: every
    stored coeffs file applies)."""
    lp = wall_cfg.get("lens_summary")
    if not lp:
        return None
    out = {}
    with open(lp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["core"]] = (r.get("lens") or "").strip()
    return out
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
def stored_params(core: str, wall_cfg=None, lenses=None):
    """coeffs.json (undistort params + crop rect) and _H.npz + canvas size
    for one image core; None if the image never made it through Stage A.

    With a lens summary (RW20C), the summary column decides whether the
    frame was undistorted: 'none'/'' -> raw (coeffs deliberately absent,
    undistort skipped); a lens name -> the coeffs file is REQUIRED and its
    absence is a hard error, never a silent raw fallback."""
    cfg = wall_cfg or WALLS["RW20"]
    corrected, rect = cfg["corrected"], cfg["rect"]
    hp = rect / f"{core}_H.npz"
    rp = rect / f"{core}_rect.png"
    if not hp.exists() or not rp.exists():
        return None
    cj = glob.glob(str(corrected / f"{core}_*coeffs.json"))
    if lenses is None:                      # RW20: coeffs always required
        if not cj:
            return None
        meta = json.loads(Path(cj[0]).read_text(encoding="utf-8"))
        und = {"xc": meta["xcenter"], "yc": meta["ycenter"],
               "coeffs": meta["coeffs_backward"],
               "crop": meta.get("out_crop")}
    else:
        lens = lenses.get(core, "")
        if lens in ("", "none"):
            und = {"xc": None, "yc": None, "coeffs": None, "crop": None}
        else:
            if not cj:
                raise RuntimeError(
                    f"{core}: summary says lens={lens} but no coeffs.json "
                    f"under {corrected} - refusing to warp a mask without "
                    f"the undistort its frame received")
            meta = json.loads(Path(cj[0]).read_text(encoding="utf-8"))
            und = {"xc": meta["xcenter"], "yc": meta["ycenter"],
                   "coeffs": meta["coeffs_backward"],
                   "crop": meta.get("out_crop")}
    d = np.load(hp, allow_pickle=True)
    rect_shape = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).shape
    return {**und, "H": d["H"], "canvas_wh": (rect_shape[1], rect_shape[0])}


def mask_to_rect(mask: np.ndarray, p: dict) -> np.ndarray:
    """Full-frame mask -> rectified raster, replaying the stored transforms
    exactly as write_corrected + auto_rectify did (float undistort, >127
    binarize, stored crop rect, NEAREST perspective warp)."""
    if p["coeffs"] is None:
        um = (mask > 127).astype(np.uint8) * 255    # frame ran raw
    else:
        import discorpy.post.postprocessing as post
        um = post.unwarp_image_backward(mask.astype(np.float32),
                                        p["xc"], p["yc"], p["coeffs"])
        um = (np.clip(um, 0, 255) > 127).astype(np.uint8) * 255
    if p["crop"]:
        cx, cy, cw, ch = p["crop"]
        um = um[cy:cy + ch, cx:cx + cw]
    return cv2.warpPerspective(um, np.asarray(p["H"], float), p["canvas_wh"],
                               flags=cv2.INTER_NEAREST)


def lift_crop_to_fullframe(pred_dir: Path, tag: str) -> Path:
    """Benchmark masks are CROP-space, not full-frame. The pool's RW20
    images are the hand-cropped frames at native resolution (verified
    2026-09-03: all 108 test images match the crop manifest w/h exactly),
    while the stored Stage-A transforms (xcenter/ycenter, out_crop) live in
    the RAW full-frame space. Feeding a crop-space mask straight into
    mask_to_rect() mis-crops silently wherever it does not crash.

    The lift is production code, reused verbatim: data/rectified/
    mask_to_full.py pastes each mask at its manifest (x, y) in the working
    frame (rotating IMG_4162 back), refuses on any dims mismatch, and its
    default --pattern already takes *_mask.png only."""
    out = BM_ROOT / tag / "fullframe"
    out.mkdir(parents=True, exist_ok=True)
    m2f = ROOT / "data" / "rectified" / "mask_to_full.py"
    # absolute paths: the subprocess runs with cwd = data/rectified (so
    # mask_to_full's `from crop_map import ...` resolves), which silently
    # re-bases any relative --masks
    cmd = [sys.executable, str(m2f), "--masks", str(pred_dir.resolve()),
           "-o", str(out.resolve())]
    print("+ " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(m2f.parent))
    if r.returncode != 0:
        raise RuntimeError("mask_to_full failed - crop-space lift is "
                           "required before the Stage-A replay")
    return out


def build_parallel_tree(pred_dir: Path, tag: str, wall: str = "RW20"):
    """Predicted masks -> parallel rectified tree. Returns (tree, coverage)."""
    cfg = WALLS[wall]
    lenses = lens_map(cfg)
    rect = cfg["rect"]
    tree = BM_ROOT / tag / "rectified"
    tree.mkdir(parents=True, exist_ok=True)
    n_in = n_ok = 0
    for f in sorted(pred_dir.iterdir()):
        # *_mask.png ONLY - infer_sam writes a matplotlib overlay as
        # <stem>.png into the same dir (the decoy the eval_masks/A1.4-30 and
        # render_overlays/A1.23-129 candidate orders exist for). This loop
        # took it too: the overlay is a ~1183px figure canvas, so the stored
        # full-res crop rect slices it empty and warpPerspective asserts.
        # Third site of the same trap; found on the first real-mask run.
        if not f.name.endswith("_mask.png"):
            continue
        core = core_of(f.name)
        if core is None:
            continue
        n_in += 1
        p = stored_params(core, cfg, lenses)
        if p is None:
            continue                       # image never rectified in Stage A
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        rm = mask_to_rect(mask, p)
        cv2.imwrite(str(tree / f"{core}_rect_mask.png"), rm)
        for side in (f"{core}_H.npz", f"{core}_rectify.json"):
            src = rect / side
            if src.exists():
                shutil.copy2(src, tree / side)
        n_ok += 1
    print(f"[front] {n_ok}/{n_in} masks entered the wall frame "
          f"(rest: no Stage-A rectification for that image)")
    return tree, (n_ok, n_in)


# ------------------------------------------------------------ chain part ---
def run_chain(tree: Path, tag: str, wall: str = "RW20") -> Path:
    fused = tree.parent / "fused"
    mech_csv = tree.parent / "mechanics_chain.csv"
    if wall == "RW20":
        # byte-for-byte the pre---wall command lines: the RW20 identity
        # gate depends on them not moving
        steps = [
            [sys.executable, str(CODE / "fuse_canvas.py"),
             "--rect", str(tree), "--out", str(fused)],
            [sys.executable, str(CODE / "link_canvas.py"),
             "--fused", str(fused)],
            [sys.executable, str(CODE / "run_mechanics_chain.py"),
             "--fused", str(fused), "--out", str(mech_csv)],
        ]
    else:
        cfg = WALLS[wall]
        steps = [
            [sys.executable, str(CODE / "fuse_canvas.py"), "--wall", wall,
             "--meta", str(cfg["meta"]),
             "--rect", str(tree), "--out", str(fused)],
            [sys.executable, str(CODE / "link_canvas.py"), "--wall", wall,
             "--fused", str(fused)],
            [sys.executable, str(CODE / "run_mechanics_chain.py"),
             "--wall", wall, "--source", "canvas",
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


def score(mech_csv: Path, tag: str, out_csv: Path, wall: str = "RW20"):
    """mechanics CSV -> x1 -> Rp errors under BOTH constants:
    - Rp_pred (K_FROZEN = 2.7902): the protocol's primary, never refit;
    - Rp_pred_fieldk (LS k on the field tier): the SS8u yardstick's own
      convention, so the 0.227 comparison stays apples-to-apples.
    Scoring basis = vision_case_study's (every step with |Rmea| >= 0.25);
    x1 == 0 rows additionally carry status NO_RESIDUAL_SIGNAL (SS8aq).

    The yardstick is the FIELD route on the same wall, computed here from
    sr_dataset's x1_field on the same scored steps (for RW20 it must land
    on the recorded 0.227 - asserted). E3 criterion: vision <= 1.10x it."""
    kf = field_k()
    field, rp_meas = {}, {}
    with open(SR_DATASET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["wall"] != wall:
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
    # field-route yardstick on the SAME wall and the SAME scored steps
    ey = [abs(law(float(r["x1_field"]), K_FROZEN) - float(r["Rmea"]))
          for r in rows
          if r["abs_err"] != "" and r["x1_field"] != ""]
    yard = sum(ey) / len(ey) if ey else float("nan")
    if wall == "RW20":
        assert abs(yard - YARDSTICK_MAE) < 1e-3,             f"RW20 field yardstick drifted: {yard:.4f} vs {YARDSTICK_MAE}"
    ratio = mae_fz / yard if yard == yard and yard else float("nan")
    print(f"[axis C] {tag} ({wall}): n={len(e_fz)} scored | frozen-k "
          f"{mae_fz:.4f} vs field yardstick {yard:.4f} = {ratio:.3f}x "
          f"(E3 bar 1.10x) | fieldk {mae_fk:.4f} | "
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
    ap.add_argument("--wall", choices=sorted(WALLS), default="RW20",
                    help="RW20 = benchmark axis C (A1.2 item 18). RW20C "
                         "runs under pipeline_rerun_spec E3 (pre-registered "
                         "there); gates stay RW20-only")
    ap.add_argument("--pred", type=Path,
                    help="model's predicted masks (*_mask.png)")
    ap.add_argument("--tag", help="e.g. a6_RW20_s0")
    ap.add_argument("--space", choices=["crop", "full"], default="crop",
                    help="coordinate space of --pred masks. Benchmark masks "
                         "are 'crop' (the pool's hand-cropped frames); "
                         "'full' is for masks already lifted to the raw "
                         "frame (preds_RW20_fullframe class). Wrong 'full' "
                         "on crop-space input mis-crops silently - the "
                         "default is therefore 'crop'.")
    args = ap.parse_args()
    if args.gate:
        if args.wall != "RW20":
            print("gates are defined against the RW20 production tree only")
            return 2
        return gate_front() if args.gate == "front" else gate_chain()
    if not (args.pred and args.tag):
        print("need --pred and --tag (or --gate front|chain)")
        return 2
    if args.wall == "RW20C" and args.space == "crop":
        print("RW20C pool masks are FULL-frame (A1.35) - pass --space full")
        return 2
    pred = args.pred
    if args.space == "crop":
        pred = lift_crop_to_fullframe(pred, args.tag)
    tree, _cov = build_parallel_tree(pred, args.tag, args.wall)
    mech = run_chain(tree, args.tag, args.wall)
    score(mech, args.tag, RESULTS / f"axis_c_{args.tag}.csv", args.wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
