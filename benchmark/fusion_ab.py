# -*- coding: utf-8 -*-
"""7.1 overlap-fusion A/B harness (docs/fusion_ab_spec.md, layer b).

Runs the FROZEN arm set on a rented box and writes one comparison table:

  A0  or   @0.25  existing rental masks (never recomputed; summary json read)
  A1  or   @0.30  \
  A2  max  @0.30   > ONE GPU pass per image: --fusion all --tile-overlap 0.30
  A3  mean @0.30  /

Per fold: infer via run_a5_zeroshot (the production shim — same argv shape,
same mask-existence check), split {stem}_mask_<mode>.png into per-arm dirs
of {stem}_mask.png, eval each arm with the single-source eval_masks, then
aggregate metrics + descriptive error-profile columns into
results/benchmark/fusion_ab.csv.

Naming rule (spec G-F3): eval outputs are fusion_eval_*.csv so
summarize_benchmark's eval_*.csv glob never sees them (the 8bk trap), and
the splitter consumes ONLY *_mask_<mode>.png — {stem}.png overlay figures
and the canonical {stem}_mask.png are decoys it must ignore (8bx class,
fourth consumer).

Decision rule (spec 5): this table is MEASUREMENT ONLY. No production
default changes here; adoption is a separate 7.4-class step (E3 reruns +
dominant-share report).

Usage (rented box, after training exists):
  python3 benchmark/fusion_ab.py run --folds RW20 RW20C
  python3 benchmark/fusion_ab.py run --folds RW20 --limit 2   # smoke
  python3 benchmark/fusion_ab.py --selftest                   # no GPU
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = HERE.parent
ARMS = ("or", "max", "mean")          # the three @0.30 arms from one pass
OVERLAP = 0.30                        # FROZEN (spec 1)
FUSION_THRESHOLD = 0.5                # FROZEN (spec 2) — never tuned here

METRIC_KEYS = ["pixel_iou", "precision", "recall", "f1",
               "cldice", "cliou_4px", "marked_fp_rate"]
FIELDS = (["fold", "arm", "fusion", "overlap", "source",
           "n_images", "n_missing_pred"] + METRIC_KEYS +
          ["fn_broken_share", "fp_isolated_share", "frag_ratio_pooled",
           "thickness_px_per_skel"])


# ------------------------------------------------------------ discovery ---

def find_ckpt(runs_dir: Path, fold: str) -> Path:
    """Same rule as axis_b's b1 default: first best_lora_weights.pt under
    the fold's a6 seed-0 run."""
    hits = sorted((runs_dir / f"a6_{fold}_s0").glob("**/best_lora_weights.pt"))
    if not hits:
        sys.exit(f"fusion_ab: FATAL - no best_lora_weights.pt under "
                 f"{runs_dir}/a6_{fold}_s0 (train the fold first)")
    return hits[0]


def resolve_marked(explicit: Path | None) -> Path:
    """Explicit resolution — the 8cj trap was a default that silently
    resolved to nothing and reported marked=0."""
    cands = ([explicit] if explicit else
             [HERE / "marked_line_images.txt",
              REPO / "marked_line_images.txt",
              REPO.parent / "Test_Crack" / "marked_line_images.txt"])
    for c in cands:
        if c and c.exists():
            return c
    sys.exit(f"fusion_ab: FATAL - marked_line_images.txt not found "
             f"(tried {[str(c) for c in cands]}). marked_fp_rate would "
             f"silently read as 'no grid-line problem' (8aq class).")


# -------------------------------------------------------------- commands ---

def build_shim_cmd(fold_dir: Path, out_all: Path, ckpt: Path,
                   config: Path, limit: int | None) -> list:
    """The exact argv for the production shim. Extracted so the selftest
    can check it with no GPU (A1.24 pattern)."""
    cmd = [sys.executable, str(HERE / "run_a5_zeroshot.py"),
           "--fold", str(fold_dir),
           "--out", str(out_all),
           "--weights", str(ckpt),
           "--config", str(config),
           "--tile-overlap", str(OVERLAP),
           "--extra", f"--fusion all --fusion-threshold {FUSION_THRESHOLD}"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def build_eval_cmd(gt_dir: Path, arm_dir: Path, out_csv: Path,
                   marked: Path) -> list:
    return [sys.executable, str(HERE / "eval_masks.py"),
            "--gt", str(gt_dir),
            "--pred", str(arm_dir),
            "--out", str(out_csv),
            "--marked-list", str(marked)]


# ----------------------------------------------------------------- split ---

def split_modes(all_dir: Path, fold_out: Path) -> dict:
    """{stem}_mask_<mode>.png -> <mode>/{stem}_mask.png.

    Consumes ONLY the per-mode files. {stem}.png (matplotlib overlay) and
    the canonical {stem}_mask.png are DECOYS and must be ignored — stems
    carry dots (IMG_x.rf.<hash>), so all name surgery is suffix string
    ops, never splitext. Hard-fails if any image lacks any mode (the spec
    guarantees one file per mode per image, all-zero included)."""
    stems_per_mode = {m: set() for m in ARMS}
    for f in sorted(all_dir.iterdir()):
        for mode in ARMS:
            suf = f"_mask_{mode}.png"
            if f.name.endswith(suf):
                stem = f.name[:-len(suf)]
                dst_dir = fold_out / mode
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst_dir / f"{stem}_mask.png")
                stems_per_mode[mode].add(stem)
                break
    all_stems = set().union(*stems_per_mode.values())
    if not all_stems:
        sys.exit(f"fusion_ab: FATAL - no *_mask_<mode>.png in {all_dir} "
                 f"(was infer run with --fusion all + --save-mask?)")
    problems = []
    for mode in ARMS:
        miss = all_stems - stems_per_mode[mode]
        if miss:
            problems.append(f"{mode} missing {sorted(miss)[:3]}"
                            f"{'...' if len(miss) > 3 else ''}")
    if problems:
        sys.exit(f"fusion_ab: FATAL - incomplete mode sets: {problems}")
    return {m: len(stems_per_mode[m]) for m in ARMS}


# ----------------------------------------------------------- aggregation ---

def profile_arm(arm_dir: Path, gts: dict) -> dict:
    """Descriptive columns (never ranked — spec 3) pooled over the arm."""
    import cv2
    from error_profile import profile_pair
    tot = dict(fn=0, fn_broken=0, fp=0, fp_iso=0,
               ncc_p=0, ncc_g=0, pred_px=0, skel=0)
    for name, gt in sorted(gts.items()):
        stem = name
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break
        pf = arm_dir / f"{stem}_mask.png"
        if pf.exists():
            pred = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            import numpy as np
            pred = np.zeros_like(gt)
        r = profile_pair(pred, gt)
        tot["fn"] += r["fn"]; tot["fn_broken"] += r["fn_broken_px"]
        tot["fp"] += r["fp"]; tot["fp_iso"] += r["fp_isolated_px"]
        tot["ncc_p"] += r["n_cc_pred"]; tot["ncc_g"] += r["n_cc_gt"]
        tot["pred_px"] += r["pred_px"]; tot["skel"] += r["skel_len_pred"]
    return {
        "fn_broken_share": (round(tot["fn_broken"] / tot["fn"], 4)
                            if tot["fn"] else ""),
        "fp_isolated_share": (round(tot["fp_iso"] / tot["fp"], 4)
                              if tot["fp"] else ""),
        "frag_ratio_pooled": (round(tot["ncc_p"] / tot["ncc_g"], 4)
                              if tot["ncc_g"] else ""),
        "thickness_px_per_skel": (round(tot["pred_px"] / tot["skel"], 4)
                                  if tot["skel"] else ""),
    }


def summary_row(fold: str, arm: str, overlap: float, source: str,
                summary: dict, prof: dict) -> dict:
    row = {"fold": fold, "arm": arm, "fusion": arm, "overlap": overlap,
           "source": source,
           "n_images": summary.get("n_images", ""),
           "n_missing_pred": summary.get("n_missing_pred", "")}
    for k in METRIC_KEYS:
        v = summary.get(k, "")
        row[k] = round(v, 4) if isinstance(v, float) else v
    row.update({k: prof.get(k, "") for k in
                ("fn_broken_share", "fp_isolated_share",
                 "frag_ratio_pooled", "thickness_px_per_skel")})
    return row


# -------------------------------------------------------------------- run ---

def cmd_run(args) -> int:
    from eval_masks import load_gt_masks
    marked = resolve_marked(args.marked_list)
    config = args.config or (REPO / "configs" / "full_lora_config.yaml")
    if not config.exists():
        sys.exit(f"fusion_ab: FATAL - config {config} not found")
    rows = []
    for fold in args.folds:
        fold_dir = args.data_root / f"fold_{fold}"
        gt_dir = fold_dir / "test"
        if not (gt_dir / "_annotations.coco.json").exists():
            sys.exit(f"fusion_ab: FATAL - {gt_dir}/_annotations.coco.json "
                     f"not found")
        ckpt = find_ckpt(args.runs_dir, fold)
        fold_out = args.masks_root / fold
        all_dir = fold_out / "all"
        all_dir.mkdir(parents=True, exist_ok=True)

        # ---- one GPU pass -> all three @0.30 arms --------------------
        cmd = build_shim_cmd(fold_dir, all_dir, ckpt, config, args.limit)
        print(f"[fusion_ab] {fold}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"fusion_ab: FATAL - infer pass failed for {fold} "
                     f"(exit {r.returncode})")
        counts = split_modes(all_dir, fold_out)
        print(f"[fusion_ab] {fold}: split {counts}", flush=True)

        gts = load_gt_masks(gt_dir)

        # ---- A0: the existing frozen rental masks (never recomputed) --
        a0 = args.results / f"eval_a6_{fold}_s0.summary.json"
        if a0.exists():
            rows.append(summary_row(
                fold, "or", 0.25, "frozen rental masks (A0)",
                json.loads(a0.read_text()), {}))
        else:
            print(f"[fusion_ab] {fold}: WARN - {a0.name} not found; "
                  f"A0 row omitted (compare against the recorded rental "
                  f"numbers instead)", flush=True)

        # ---- eval + profile each @0.30 arm ---------------------------
        for mode in ARMS:
            arm_dir = fold_out / mode
            out_csv = args.results / f"fusion_eval_{fold}_{mode}.csv"
            ev = subprocess.run(build_eval_cmd(gt_dir, arm_dir, out_csv,
                                               marked))
            if ev.returncode != 0:
                sys.exit(f"fusion_ab: FATAL - eval failed for "
                         f"{fold}/{mode}")
            summ = json.loads(
                out_csv.with_suffix(".summary.json").read_text())
            prof = profile_arm(arm_dir, gts)
            rows.append(summary_row(fold, mode, OVERLAP,
                                    "one-pass --fusion all", summ, prof))

    out = args.out or (args.results / "fusion_ab.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"fusion_ab -> {out} ({len(rows)} rows) "
          f"[MEASUREMENT ONLY: adoption is a separate 7.4-class step, "
          f"spec 5]")
    return 0


# --------------------------------------------------------------- selftest ---

def selftest() -> int:
    import fnmatch
    import tempfile

    # 1) shim command assembly (spec G-F3)
    cmd = build_shim_cmd(Path("data/fold_RW20"), Path("m/RW20/all"),
                         Path("runs/a6_RW20_s0/best_lora_weights.pt"),
                         Path("configs/full_lora_config.yaml"), None)
    ex = cmd[cmd.index("--extra") + 1]
    assert "--fusion all" in ex, f"--fusion all missing from extra: {ex}"
    assert f"--fusion-threshold {FUSION_THRESHOLD}" in ex, \
        "frozen threshold not pinned in argv"
    assert cmd[cmd.index("--tile-overlap") + 1] == "0.3", \
        "frozen overlap 0.30 not in argv"
    assert "--weights" in cmd, "a6 checkpoint must be passed (never a5 here)"
    smoke = build_shim_cmd(Path("f"), Path("o"), Path("w"), Path("c"), 2)
    assert smoke[smoke.index("--limit") + 1] == "2"

    # 2) eval output naming must dodge summarize's eval_*.csv glob (8bk)
    name = "fusion_eval_RW20_max.csv"
    assert not fnmatch.fnmatch(name, "eval_*.csv"), \
        "fusion eval output would be swept up by summarize_benchmark"
    ecmd = build_eval_cmd(Path("gt"), Path("p"), Path(name), Path("mk"))
    assert "--marked-list" in ecmd, "marked list must be explicit (8cj trap)"

    # 3) splitter on planted files: dotted stems, decoys ignored,
    #    missing-mode detected
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        alld = root / "all"; alld.mkdir()
        stem = "IMG_4100_JPG_JPG.rf.dadc3afacbaac24cafbceb75c1b783c6"
        for mode in ARMS:
            (alld / f"{stem}_mask_{mode}.png").write_bytes(b"png")
        (alld / f"{stem}.png").write_bytes(b"overlay-decoy")
        (alld / f"{stem}_mask.png").write_bytes(b"canonical-decoy")
        counts = split_modes(alld, root)
        assert counts == {m: 1 for m in ARMS}, counts
        for mode in ARMS:
            dst = root / mode / f"{stem}_mask.png"
            assert dst.exists(), f"{mode} split output missing"
            assert dst.read_bytes() == b"png", \
                f"{mode}: a decoy leaked into the arm dir"
        # NEGATIVE: drop one mode file for a second image -> must exit 1
        stem2 = "IMG_0002.rf.ffff"
        for mode in ("or", "max"):
            (alld / f"{stem2}_mask_{mode}.png").write_bytes(b"png")
        try:
            split_modes(alld, root / "again")
        except SystemExit as e:
            assert "incomplete mode sets" in str(e.code), e.code
        else:
            raise AssertionError(
                "splitter accepted an image missing the mean mask - the "
                "one-file-per-mode guarantee is unenforced")

    # 4) row schema stays aligned with the CSV header
    row = summary_row("RW20", "max", OVERLAP, "test",
                      {"cldice": 0.91234, "n_images": 3},
                      {"frag_ratio_pooled": 1.5})
    assert set(row) <= set(FIELDS), set(row) - set(FIELDS)
    assert row["cldice"] == 0.9123

    print("fusion_ab selftest PASS: argv pins fusion-all/threshold/overlap, "
          "fusion_eval_* dodges the eval_* glob, splitter ignores both "
          "decoys and hard-fails on a missing mode")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--selftest", action="store_true")
    rp = sub.add_parser("run")
    rp.add_argument("--folds", nargs="+", required=True)
    rp.add_argument("--data-root", type=Path, default=Path("data"))
    rp.add_argument("--runs-dir", type=Path, default=Path("runs"))
    rp.add_argument("--results", type=Path, default=Path("results/benchmark"))
    rp.add_argument("--masks-root", type=Path,
                    default=Path("results/benchmark/masks_fusion"))
    rp.add_argument("--config", type=Path, default=None)
    rp.add_argument("--marked-list", type=Path, default=None)
    rp.add_argument("--limit", type=int, default=None,
                    help="first N test images only (smoke)")
    rp.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cmd == "run":
        return cmd_run(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
