# -*- coding: utf-8 -*-
"""Benchmark figures from the numeric CSVs — house pattern of
code/make_report_figs.py: pure CSV reader -> PNG, never computes results.

The CSVs are the source of truth (Amendment A1.1 item 17); these figures
are conveniences — the user can always re-plot with different axes from
the same files.

Reads results/benchmark/: plot_data_long.csv, main_table.csv,
per_image_metrics.csv, a5_vs_a6.csv (+ runs/<tag>/{train,valid}_log.csv
for curves). Writes PNGs into --out (default results/benchmark/figs).

  python make_bench_figs.py [--results results/benchmark] [--runs-dir runs]
  python make_bench_figs.py --selftest    # runs on the summarizer selftest CSVs
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["pixel_iou", "f1", "cldice", "cliou_4px"]


def rd(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_pooled_bar(results: Path, out: Path):
    rows = [r for r in rd(results / "main_table.csv")
            if r["scope"] == "POOLED" and r["metric"] in METRICS]
    if not rows:
        return False
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4))
    for ax, metric in zip(axes, METRICS):
        ms, med, lo, hi = [], [], [], []
        for m in models:
            r = next((r for r in rows if r["model"] == m
                      and r["metric"] == metric), None)
            if r is None:
                continue
            ms.append(m)
            med.append(float(r["median"]))
            lo.append(float(r["median"]) - float(r["min"]))
            hi.append(float(r["max"]) - float(r["median"]))
        ax.bar(ms, med, yerr=[lo, hi], capsize=4, color="#4878a8")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("POOLED (counts over 4 test walls), median with seed range")
    fig.tight_layout()
    fig.savefig(out / "fig_pooled_bar.png", dpi=150)
    plt.close(fig)
    return True


def fig_fold_heatmap(results: Path, out: Path, metric="cliou_4px"):
    rows = [r for r in rd(results / "plot_data_long.csv")
            if r["metric"] == metric and r["fold"] != "POOLED"]
    if not rows:
        return False
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["fold"])].append(float(r["value"]))
    models = sorted({m for m, _ in agg})
    folds = sorted({f for _, f in agg})
    import numpy as np
    grid = np.full((len(models), len(folds)), np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(folds):
            v = agg.get((m, f))
            if v:
                v = sorted(v)
                grid[i, j] = v[len(v) // 2]
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(folds),
                                    1.5 + 0.6 * len(models)))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(folds)), folds)
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(folds)):
            if grid[i, j] == grid[i, j]:
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, label=metric)
    ax.set_title(f"{metric} per fold (median over seeds)")
    fig.tight_layout()
    fig.savefig(out / f"fig_fold_heatmap_{metric}.png", dpi=150)
    plt.close(fig)
    return True


def fig_per_image_box(results: Path, out: Path, metric="cliou_4px"):
    rows = rd(results / "per_image_metrics.csv")
    if not rows:
        return False
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r[metric]))
    models = sorted(by_model)
    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(models), 4.5))
    ax.boxplot([by_model[m] for m in models], tick_labels=models)
    ax.set_ylabel(metric)
    ax.set_title(f"per-image {metric} distribution (all folds/seeds)")
    fig.tight_layout()
    fig.savefig(out / f"fig_per_image_box_{metric}.png", dpi=150)
    plt.close(fig)
    return True


def fig_delta_a5_a6(results: Path, out: Path):
    rows = [r for r in rd(results / "a5_vs_a6.csv") if r.get("delta")]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["metric"] for r in rows], [float(r["delta"]) for r in rows],
           color="#5a9e6f")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("A6 − A5")
    ax.set_title("LoRA + domain-training contribution")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_a5_a6.png", dpi=150)
    plt.close(fig)
    return True


def fig_curves(runs_dir: Path, out: Path):
    made = 0
    for run in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        tl = rd(run / "train_log.csv")
        vl = rd(run / "valid_log.csv")
        if not tl:
            continue
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot([int(r["step"]) for r in tl],
                 [float(r["total"]) for r in tl], lw=0.8, label="train loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        if vl:
            ax2 = ax1.twinx()
            ax2.plot([int(r["step"]) for r in vl],
                     [float(r["valid_score"]) for r in vl], "o-",
                     color="#c05540", label="valid score")
            ax2.set_ylabel("valid (hard-dice proxy)")
        ax1.set_title(run.name)
        fig.tight_layout()
        fig.savefig(out / f"fig_curve_{run.name}.png", dpi=130)
        plt.close(fig)
        made += 1
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/benchmark"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    out = args.out or (args.results / "figs")
    out.mkdir(parents=True, exist_ok=True)
    done = {
        "pooled_bar": fig_pooled_bar(args.results, out),
        "fold_heatmap": fig_fold_heatmap(args.results, out),
        "per_image_box": fig_per_image_box(args.results, out),
        "delta_a5_a6": fig_delta_a5_a6(args.results, out),
        "curves": fig_curves(args.runs_dir, out),
    }
    print({k: v for k, v in done.items()})
    print(f"figs -> {out}")
    return 0


def selftest():
    """Generate the summarizer's synthetic CSVs, then every figure must
    render without error and produce a nonzero PNG."""
    import subprocess
    import sys
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    here = Path(__file__).resolve().parent
    # reuse the summarizer selftest machinery to fabricate a results dir
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import summarize_benchmark as sb\n"
        "from pathlib import Path\n"
        "import json, csv\n"
        "from eval_masks import finalize\n"
        "res = Path(r'%s')\n"
        "res.mkdir(exist_ok=True)\n"
        "for model, tp in (('a5', 50), ('a6', 80)):\n"
        "    for fold in ('F1', 'F2'):\n"
        "        for sd in ((None,) if model=='a5' else (0,1)):\n"
        "            acc = dict(tp=tp, fp=10, fn=10, sp_in_g=tp, sp=tp+10,\n"
        "                       sg_in_p=tp, sg=tp+10, cl_tp=tp, cl_fp=10,\n"
        "                       cl_fn=10, marked_fp=0, marked_pixels=0)\n"
        "            s = finalize(acc) | {'counts': acc, 'n_images': 1,\n"
        "                                 'n_missing_pred': 0}\n"
        "            tag = f'eval_{model}_{fold}' + ('' if sd is None else f'_s{sd}')\n"
        "            (res/f'{tag}.summary.json').write_text(json.dumps(s))\n"
        "            with open(res/f'{tag}.csv', 'w', newline='') as fh:\n"
        "                w = csv.DictWriter(fh, fieldnames=['image','px']+sb.PER_IMAGE_COUNTS)\n"
        "                w.writeheader()\n"
        "                w.writerow({'image':'i.jpg','px':4096, **{k: acc[k] for k in sb.PER_IMAGE_COUNTS}})\n"
        "sb.MODELS = [('a5','A5'), ('a6','A6')]; sb.SEEDLESS = {'a5'}\n"
        "sb.main(['--results', str(res), '--folds', 'F1', 'F2', '--seeds', '0', '1'])\n"
    ) % (here, tmp)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rc = main(["--results", str(tmp), "--runs-dir", str(tmp / "no_runs")])
    assert rc == 0
    made = list((tmp / "figs").glob("*.png"))
    assert len(made) >= 4, made
    assert all(p.stat().st_size > 5000 for p in made), \
        [(p.name, p.stat().st_size) for p in made]
    print(f"selftest PASS: {len(made)} figures rendered from synthetic CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
