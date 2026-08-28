# -*- coding: utf-8 -*-
"""Aggregate all eval_<tag>.summary.json into the paper tables — the
"one script reads *.summary.json" step promised in vastai_runbook.md SS5.

Runs automatically as the LAST queue job (make_jobs.py appends it), so the
instance finishes with paper-ready tables before --poweroff; rerunnable
locally on the returned files, bit-identical.

Outputs (into --results):
  main_table.csv / main_table.md   rows A1-A6 x (per fold + POOLED);
                                   columns per Amendment A1 order:
                                   pixel IoU | F1 | clDice | clIoU_4px |
                                   marked-FP% | params | ms/tile
  a5_vs_a6.csv                     LoRA contribution (delta per metric)
  seed_variance.csv                range per (model, fold, metric)
  summary_all.json                 every number, for downstream scripts

Pooling: LOWO test sets are disjoint walls, so POOLED = sum the raw counts
across the 4 folds (per seed) and push them through eval_masks.finalize()
— the SAME function that scored each fold; there is no second metric
definition. Ratios are never averaged. Seeds are then reported as
median (min-max) — never a mean that hides spread.

House rule (no-silent-caps): a missing run never drops silently — the cell
prints INCOMPLETE and --strict exits 1 (the queue job uses --strict, so an
incomplete grid fails loudly in queue_state.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_masks import finalize  # noqa: E402  (single metric definition)

# display order = benchmark_protocol.md rows
MODELS = [
    ("a1",         "A1 nnU-Net"),
    ("unet",       "A2 U-Net"),
    ("deeplabv3p", "A3 DeepLabv3+"),
    ("segformer",  "A4 SegFormer-B2"),
    ("a5",         "A5 SAM3 zero-shot"),
    ("a6",         "A6 SAM3-LoRA"),
]
SEEDLESS = {"a1", "a5"}          # a5 deterministic; a1 seed policy = A1.1 item 13
METRICS = ["pixel_iou", "f1", "cldice", "cliou_4px", "marked_fp_rate"]
HEADERS = ["pixel IoU", "F1", "clDice", "clIoU_4px", "marked-FP%"]
COUNT_KEYS = ["tp", "fp", "fn", "sp_in_g", "sp", "sg_in_p", "sg",
              "cl_tp", "cl_fp", "cl_fn", "marked_fp", "marked_pixels"]
TAG_RE = re.compile(r"^eval_([a-z0-9]+)_([A-Za-z0-9]+?)(?:_s(\d+))?$")


def discover(results: Path) -> dict:
    """{(model, fold, seed|None): summary dict} from eval_*.summary.json."""
    runs = {}
    for f in sorted(results.glob("eval_*.summary.json")):
        m = TAG_RE.match(f.stem.replace(".summary", ""))
        if not m:
            print(f"WARN: unrecognized tag {f.name} — skipped (loudly)")
            continue
        model, fold, seed = m.group(1), m.group(2), m.group(3)
        runs[(model, fold, int(seed) if seed is not None else None)] = \
            json.loads(f.read_text(encoding="utf-8"))
    return runs


def pool(summaries: list[dict]) -> dict:
    """Sum raw counts across folds -> finalize() (exact, LOWO-disjoint)."""
    acc = {k: 0 for k in COUNT_KEYS}
    for s in summaries:
        c = s.get("counts")
        if c is None:
            raise KeyError("summary has no 'counts' — re-run eval_masks "
                           "(counts added 2026-08-28) before pooling")
        for k in COUNT_KEYS:
            acc[k] += c.get(k, 0)
    return finalize(acc)


def med_range(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.median(vals):.4f} ({min(vals):.4f}-{max(vals):.4f})"


def pct(v):
    return "—" if v is None else f"{100 * v:.3f}%"


def run_extras(model: str, fold: str, seed, runs_dir: Path):
    """params (run.json) + ms/tile (predict_run.json) — TBD when absent."""
    tag = f"{model}_{fold}" + (f"_s{seed}" if seed is not None else "")
    params = ms = None
    rj = runs_dir / tag / "run.json"
    if rj.exists():
        params = json.loads(rj.read_text()).get("params")
    pj = runs_dir / tag / "masks" / "predict_run.json"
    if pj.exists():
        ms = json.loads(pj.read_text()).get("ms_per_tile")
    return params, ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/benchmark"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--folds", nargs="+",
                    default=["RW20", "RW20C", "RW20L", "RW20T"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any expected run is missing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    runs = discover(args.results)
    if not runs:
        print(f"no eval_*.summary.json under {args.results}")
        return 1

    incomplete = []
    table_rows = []          # main_table.csv rows
    md = ["| model | scope | " + " | ".join(HEADERS)
          + " | params | ms/tile |",
          "|---|---|" + "---|" * (len(HEADERS) + 2)]
    all_out = {"folds": args.folds, "seeds": args.seeds, "models": {}}
    seed_var_rows = []
    pooled_by_model = {}     # model -> {metric: median-over-seeds float}

    for model, label in MODELS:
        seeds = [None] if model in SEEDLESS else args.seeds
        mrec = {"label": label, "per_fold": {}, "pooled": {}}

        # --- per fold ---------------------------------------------------
        for fold in args.folds:
            per_seed = {}
            for sd in seeds:
                s = runs.get((model, fold, sd))
                if s is None and sd is None:
                    # tolerate a1_RW20_s0-style names for seedless rows
                    s = runs.get((model, fold, 0))
                if s is None:
                    incomplete.append(f"{model}_{fold}"
                                      + (f"_s{sd}" if sd is not None else ""))
                    continue
                per_seed[str(sd)] = {m: s.get(m) for m in METRICS}
            mrec["per_fold"][fold] = per_seed
            for metric in METRICS:
                vals = [v[metric] for v in per_seed.values()
                        if v.get(metric) is not None]
                if len(vals) > 1:
                    seed_var_rows.append({
                        "model": model, "fold": fold, "metric": metric,
                        "median": statistics.median(vals),
                        "min": min(vals), "max": max(vals),
                        "range": max(vals) - min(vals), "n_seeds": len(vals)})

        # --- pooled across folds (per seed), then median over seeds -----
        pooled_seeds = {}
        for sd in seeds:
            ss = [runs.get((model, fold, sd)) or
                  (runs.get((model, fold, 0)) if sd is None else None)
                  for fold in args.folds]
            if all(s is not None and "counts" in s for s in ss):
                pooled_seeds[str(sd)] = pool(ss)
        mrec["pooled"] = pooled_seeds
        all_out["models"][model] = mrec

        n_expected = len(args.folds) * len(seeds)
        n_have = sum(len(v) for v in mrec["per_fold"].values())
        status = "" if n_have == n_expected else \
            f"  INCOMPLETE ({n_have}/{n_expected} runs)"

        # extras: median over the runs that have them
        pvals, msvals = [], []
        for fold in args.folds:
            for sd in seeds:
                p, m = run_extras(model, fold, sd, args.runs_dir)
                if p is not None:
                    pvals.append(p)
                if m is not None:
                    msvals.append(m)
        p_str = f"{statistics.median(pvals) / 1e6:.1f}M" if pvals else "TBD"
        ms_str = f"{statistics.median(msvals):.1f}" if msvals else "TBD"

        # rows: per fold then POOLED
        for fold in args.folds:
            per_seed = mrec["per_fold"][fold]
            cells = []
            for metric in METRICS:
                vals = [v[metric] for v in per_seed.values()
                        if v.get(metric) is not None]
                cells.append(med_range(vals))
            table_rows.append({"model": model, "scope": fold,
                               **dict(zip(METRICS, cells))})
        pcells = []
        pooled_med = {}
        for metric in METRICS:
            vals = [p.get(metric) for p in pooled_seeds.values()
                    if p.get(metric) is not None]
            pcells.append(med_range(vals))
            if vals:
                pooled_med[metric] = statistics.median(vals)
        pooled_by_model[model] = pooled_med
        table_rows.append({"model": model, "scope": "POOLED" + status,
                           **dict(zip(METRICS, pcells)),
                           "params": p_str, "ms_per_tile": ms_str})
        md.append(f"| {label} | POOLED{status} | " + " | ".join(pcells)
                  + f" | {p_str} | {ms_str} |")

    # --- A5 vs A6 (LoRA contribution) -----------------------------------
    a5, a6 = pooled_by_model.get("a5", {}), pooled_by_model.get("a6", {})
    delta_rows = [{"metric": m, "a5": a5.get(m), "a6": a6.get(m),
                   "delta": (a6[m] - a5[m]) if m in a5 and m in a6 else None}
                  for m in METRICS]

    # --- write everything ------------------------------------------------
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    fields = ["model", "scope"] + METRICS + ["params", "ms_per_tile"]
    with open(out / "main_table.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)
    (out / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "a5_vs_a6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "a5", "a6", "delta"])
        w.writeheader()
        w.writerows(delta_rows)
    with open(out / "seed_variance.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fold", "metric", "median",
                                          "min", "max", "range", "n_seeds"])
        w.writeheader()
        w.writerows(seed_var_rows)
    all_out["incomplete"] = incomplete
    (out / "summary_all.json").write_text(json.dumps(all_out, indent=2),
                                          encoding="utf-8")

    print("\n".join(md))
    if incomplete:
        print(f"\nINCOMPLETE — {len(incomplete)} expected runs missing:")
        for t in incomplete:
            print(f"  {t}")
        if args.strict:
            return 1
    else:
        print("\nall expected runs present")
    print(f"wrote main_table.csv/.md, a5_vs_a6.csv, seed_variance.csv, "
          f"summary_all.json -> {out}")
    return 0


# --------------------------------------------------------------- selftest --
def selftest():
    """Planted counts must reproduce through pooling + table, and a missing
    run must surface as INCOMPLETE with --strict exit 1."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = tmp / "results"
    res.mkdir()
    folds, seeds = ["F1", "F2"], [0, 1]

    def write(model, fold, seed, tp, fp, fn):
        acc = dict(tp=tp, fp=fp, fn=fn, sp_in_g=tp, sp=tp + fp,
                   sg_in_p=tp, sg=tp + fn, cl_tp=tp, cl_fp=fp, cl_fn=fn,
                   marked_fp=0, marked_pixels=0)
        s = finalize(acc) | {"counts": acc, "n_images": 1,
                             "n_missing_pred": 0}
        tag = f"eval_{model}_{fold}" + (f"_s{seed}" if seed is not None
                                        else "")
        (res / f"{tag}.summary.json").write_text(json.dumps(s))

    # a6: seeds x folds, counts chosen so pooled is hand-checkable
    for sd in seeds:
        write("a6", "F1", sd, tp=80, fp=10, fn=10)   # iou 0.8
        write("a6", "F2", sd, tp=60, fp=20, fn=20)   # iou 0.6
    write("a5", "F1", None, tp=50, fp=25, fn=25)
    write("a5", "F2", None, tp=50, fp=25, fn=25)     # iou 0.5 each

    global MODELS, SEEDLESS
    keep_models, keep_seedless = MODELS, SEEDLESS
    MODELS = [("a5", "A5"), ("a6", "A6")]
    SEEDLESS = {"a5"}
    try:
        rc = main(["--results", str(res), "--folds", *folds,
                   "--seeds", *map(str, seeds)])
        assert rc == 0, rc
        allj = json.loads((res / "summary_all.json").read_text())
        pooled = allj["models"]["a6"]["pooled"]["0"]["pixel_iou"]
        expect = (80 + 60) / (80 + 60 + 10 + 20 + 10 + 20)   # 140/200 = 0.7
        assert abs(pooled - expect) < 1e-12, (pooled, expect)
        a5p = allj["models"]["a5"]["pooled"]["None"]["pixel_iou"]
        assert abs(a5p - 0.5) < 1e-12, a5p
        rows = list(csv.DictReader(open(res / "a5_vs_a6.csv")))
        d = next(r for r in rows if r["metric"] == "pixel_iou")
        assert abs(float(d["delta"]) - 0.2) < 1e-12, d

        # missing-run case: drop one seed -> INCOMPLETE + strict exit 1
        (res / "eval_a6_F2_s1.summary.json").unlink()
        rc = main(["--results", str(res), "--folds", *folds,
                   "--seeds", *map(str, seeds), "--strict"])
        assert rc == 1, "strict must fail on a missing run"
        allj = json.loads((res / "summary_all.json").read_text())
        assert "a6_F2_s1" in allj["incomplete"], allj["incomplete"]
    finally:
        MODELS, SEEDLESS = keep_models, keep_seedless
    print("selftest PASS: pooled=counts-exact (0.7/0.5), delta 0.2, "
          "missing run -> INCOMPLETE + strict exit 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
