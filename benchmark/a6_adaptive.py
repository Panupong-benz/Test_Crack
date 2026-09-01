# -*- coding: utf-8 -*-
"""a6_adaptive - the pre-registered UPWARD-ONLY budget rule for row A6, as three
queue jobs that keep jobs.yaml static (Amendment A1.22 items 117-121).

    decide  --pilot-run runs/a6_RW20_s0 --base 30 --ext 60
        Judges the pilot's B-epoch curve with the FROZEN A1.7 criterion
        (same code path as the end-of-queue epoch_saturation job) and writes
        results/benchmark/epoch_budget_decision.{csv,md}:
            budget_limited                  -> chosen = ext (whole row)
            saturated / degenerate          -> chosen = base
            no_log / restarts>0 / partial   -> exit 1 (something is broken)
    extend  --run runs/a6_RW20_s0 --base-config ... --config-dir ...
        No-op (exit 0) when the decision is `base`. Otherwise copies the
        B-epoch artifacts to results/benchmark/pre_extension/, writes
        configs/benchmark/<tag>_e<ext>.yaml (base config + the SAME
        override_keys as make_jobs + num_epochs=ext AFTER the walk), launches
        the trainer with --resume, and touches runs/<tag>/EXTENDED on success.
    train   --fold RW20C --seed 0 ...
        Later folds of the row: reads the decision file (HARD FAIL if absent -
        never a silent default), writes the fold's config at the chosen budget,
        launches the trainer with --resume.

Why static jobs and not branching: queue_runner runs a fixed list and skips
only names marked ok; every branch above is a job that exits 0 quickly when
its branch is not taken. --dry-run prints the trainer command instead of
launching it (used by the selftest).

The rule is written here AND in benchmark_protocol.md A1.22; changing either
requires its own commit stating what changed and why.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import epoch_saturation as es          # noqa: E402  (frozen criterion)
from make_jobs import override_keys    # noqa: E402  (identical config path)

DECISION = "epoch_budget_decision"
RULE = ("pilot budget_limited (A1.7 frozen criterion) -> whole row at ext; "
        "saturated/degenerate -> base; unreadable -> error")


def _tag(run: Path) -> str:
    return Path(run).name


def _parse_tag(tag: str):
    m = es.TAG_RE.match(tag)
    if not m or m.group(1) != "a6":
        raise SystemExit(f"a6_adaptive: not an a6 run dir name: {tag}")
    return m.group(2), int(m.group(3) or 0)


# ------------------------------------------------------------------ decide --
def cmd_decide(a) -> int:
    run = Path(a.pilot_run)
    fold, seed = _parse_tag(_tag(run))
    brow = es.budget_row(run, "a6", Path(a.config_dir), None)
    budget = brow.get("epochs")
    vals, meta = es.read_val_stats(run)
    vals = vals or []
    problems = []
    if not isinstance(budget, int) or budget <= 0:
        problems.append("budget unknown (no run.json / config)")
    elif budget != a.base:
        problems.append(f"pilot budget {budget} != --base {a.base}")
    if int(meta.get("restarts", 0)) > 0:
        problems.append(f"curve has {meta['restarts']} restart block(s)")
    if not vals:
        problems.append("no val_stats.json / no_log")
    elif len(vals) != a.base:
        problems.append(f"partial curve: {len(vals)} of {a.base} epochs logged")
    if problems:
        print("[decide] REFUSED - " + "; ".join(problems))
        return 1
    r = es.judge(vals, a.base, higher_is_better=False)
    chosen = a.ext if r["verdict"] == "budget_limited" else a.base
    row = {"pilot_run": _tag(run), "fold": fold, "seed": seed,
           "base": a.base, "ext": a.ext, "verdict": r["verdict"],
           "best_epoch": r["best_epoch"], "best_frac": r["best_frac"],
           "rel_tail_gain": r["rel_tail_gain"], "epochs_logged": r["epochs_logged"],
           "chosen_budget": chosen, "rule": RULE}
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{DECISION}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    md = [f"# Epoch-budget decision (A1.22) - pilot {row['pilot_run']}", "",
          f"- rule: {RULE}",
          f"- pilot curve: {r['epochs_logged']} epochs at B = {a.base}, "
          f"best epoch **{r['best_epoch']}** (best_frac {r['best_frac']}), "
          f"rel_tail_gain {r['rel_tail_gain']}",
          f"- verdict under the frozen criterion: **{r['verdict']}**",
          f"- **chosen budget for the whole row: {chosen} epochs**"
          + (" (extension: pilot continues via --resume; later folds train at "
             f"{a.ext} from scratch)" if chosen == a.ext else
             " (no extension)"),
          "- full curve (val_loss per epoch): " + ", ".join(f"{v:.3f}" for v in vals),
          ]
    (out / f"{DECISION}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[decide] {row['pilot_run']}: verdict={r['verdict']} best_epoch="
          f"{r['best_epoch']} -> chosen budget {chosen}")
    return 0


def _read_decision(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"a6_adaptive: {path} missing - the pilot's `decide` job "
                         f"has not run; refusing to guess a budget")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if len(rows) != 1:
        raise SystemExit(f"a6_adaptive: {path} must hold exactly one row")
    return rows[0]


def _write_cfg(base_config: Path, data_root: str, fold: str, seed: int,
               epochs: int, cfg_path: Path, gpus: int = 1):
    import yaml
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    tag = f"a6_{fold}_s{seed}"
    cfg = override_keys(base, f"{data_root}/fold_{fold}", seed, f"runs/{tag}",
                        gpus=gpus)
    cfg["training"]["num_epochs"] = int(epochs)   # AFTER the walk (A1.21 item 114)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def _dev(gpus: int) -> list:
    """--device 0..N-1 for N>1 (trainer self-launches torchrun); [] for 1 so
    single-GPU commands are byte-identical to pre-A1.27 (A1.27 item 152(d))."""
    return (["--device"] + [str(i) for i in range(gpus)]) if gpus > 1 else []


def _launch(cmd, dry_run: bool) -> int:
    print("[a6_adaptive] " + ("DRY-RUN: " if dry_run else "launch: ")
          + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"))


# ------------------------------------------------------------------ extend --
def cmd_extend(a) -> int:
    run = Path(a.run)
    tag = _tag(run)
    fold, seed = _parse_tag(tag)
    d = _read_decision(Path(a.results) / f"{DECISION}.csv")
    chosen, base = int(d["chosen_budget"]), int(d["base"])
    if chosen == base:
        print(f"[extend] decision = {base} (verdict {d['verdict']}): no extension - no-op")
        return 0
    # keep the B-epoch evaluation auditable before it is overwritten
    pre = Path(a.results) / "pre_extension"
    pre.mkdir(parents=True, exist_ok=True)
    for src in (Path(a.results) / f"eval_{tag}.csv",
                Path(a.results) / f"eval_{tag}.summary.json",
                run / "masks" / "a5_run.json"):
        if src.exists():
            shutil.copy2(src, pre / (f"{tag}__" + src.name))
    if (run / "val_stats.json").exists():
        shutil.copy2(run / "val_stats.json", run / f"val_stats.B{base}.json")
    if (run / "best_lora_weights.pt").exists():
        shutil.copy2(run / "best_lora_weights.pt", run / f"best_lora_weights.B{base}.pt")
    cfg_path = Path(a.config_dir) / f"{tag}_e{chosen}.yaml"
    _write_cfg(Path(a.base_config), a.data_root, fold, seed, chosen, cfg_path,
               gpus=getattr(a, "gpus", 1))
    print(f"[extend] {tag}: {base} -> {chosen} epochs via --resume "
          f"(config {cfg_path})")
    rc = _launch([sys.executable if a.dry_run else "python3", a.trainer,
                  "--config", cfg_path.as_posix(), "--resume"] + _dev(getattr(a, "gpus", 1)),
                 a.dry_run)
    if rc == 0 and not a.dry_run:
        (run / "EXTENDED").write_text(f"{base}->{chosen}\n")
    return rc


# ------------------------------------------------------------------- train --
def cmd_train(a) -> int:
    d = _read_decision(Path(a.results) / f"{DECISION}.csv")
    chosen = int(d["chosen_budget"])
    tag = f"a6_{a.fold}_s{a.seed}"
    cfg_path = Path(a.config_dir) / f"{tag}.yaml"
    _write_cfg(Path(a.base_config), a.data_root, a.fold, a.seed, chosen, cfg_path,
               gpus=getattr(a, "gpus", 1))
    print(f"[train] {tag} at the row's decided budget {chosen} "
          f"(pilot {d['pilot_run']} verdict {d['verdict']})")
    return _launch([sys.executable if a.dry_run else "python3", a.trainer,
                    "--config", cfg_path.as_posix(), "--resume"] + _dev(getattr(a, "gpus", 1)),
                   a.dry_run)


# ---------------------------------------------------------------- selftest --
def selftest() -> int:
    import tempfile
    import yaml
    base_cfg = {"training": {"num_epochs": 30, "data_dir": "x", "seed": 42,
                             "batch_size": 2, "gradient_accumulation_steps": 8},
                "lora": {"rank": 16, "alpha": 32},
                "output": {"output_dir": "x"}}

    def curve(best_at, n=30):
        return [300 * 0.82 ** min(i, best_at) + 40 for i in range(1, n + 1)]

    def plant(root, vals, epochs=30):
        d = root / "runs" / "a6_F1_s0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "val_stats.json").write_text("".join(
            json.dumps({"epoch": i, "val_loss": v}) + "\n"
            for i, v in enumerate(vals, 1)))
        (d / "run.json").write_text(json.dumps(
            {"epochs": epochs, "batch": 2, "grad_accum": 8, "n_train_tiles": 6000}))
        return d

    def args(**kw):
        return argparse.Namespace(**kw)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bc = root / "base.yaml"
        bc.write_text(yaml.safe_dump(base_cfg))
        common = dict(results=str(root / "res"), config_dir=str(root / "cfg"),
                      base_config=str(bc), data_root="data", trainer="T.py",
                      dry_run=True)

        # 1. saturated pilot -> base, extend is a no-op
        d = plant(root, curve(12))
        assert cmd_decide(args(pilot_run=str(d), base=30, ext=60, **common)) == 0
        dec = _read_decision(root / "res" / f"{DECISION}.csv")
        assert dec["verdict"] == "saturated" and dec["chosen_budget"] == "30", dec
        assert cmd_extend(args(run=str(d), **common)) == 0
        assert not (root / "cfg" / "a6_F1_s0_e60.yaml").exists()
        print("  saturated pilot -> 30, extend no-op")

        # 2. budget-limited pilot -> 60; extend writes e60 yaml differing ONLY in
        #    num_epochs; train for the next fold uses 60
        climb = [300 - 8 * i for i in range(1, 31)]          # still falling at 30
        d = plant(root, climb)
        assert cmd_decide(args(pilot_run=str(d), base=30, ext=60, **common)) == 0
        dec = _read_decision(root / "res" / f"{DECISION}.csv")
        assert dec["verdict"] == "budget_limited" and dec["chosen_budget"] == "60", dec
        assert cmd_extend(args(run=str(d), **common)) == 0
        e60 = yaml.safe_load((root / "cfg" / "a6_F1_s0_e60.yaml").read_text())
        ref = override_keys(base_cfg, "data/fold_F1", 0, "runs/a6_F1_s0")
        ref["training"]["num_epochs"] = 60
        assert e60 == ref, (e60, ref)
        assert cmd_train(args(fold="F2", seed=0, **common)) == 0
        t2 = yaml.safe_load((root / "cfg" / "a6_F2_s0.yaml").read_text())
        assert t2["training"]["num_epochs"] == 60 and t2["training"]["data_dir"] == "data/fold_F2"
        md = (root / "res" / f"{DECISION}.md").read_text(encoding="utf-8")
        assert "budget_limited" in md and "60" in md
        print("  budget_limited pilot -> 60, e60 config differs only in num_epochs, next fold at 60")

        # 3. refusals: partial curve, restarted curve, missing decision
        d = plant(root, curve(12)[:20])
        assert cmd_decide(args(pilot_run=str(d), base=30, ext=60, **common)) == 1
        d = plant(root, curve(5, 9) + curve(12))
        assert cmd_decide(args(pilot_run=str(d), base=30, ext=60, **common)) == 1
        (root / "res" / f"{DECISION}.csv").unlink()
        try:
            cmd_train(args(fold="F2", seed=0, **common))
            raise AssertionError("train without a decision must fail")
        except SystemExit as e:
            assert "missing" in str(e)
        print("  refusals: partial curve / restarted curve / missing decision")
    print("selftest PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--results", default="results/benchmark")
        sp.add_argument("--config-dir", default="configs/benchmark")
        sp.add_argument("--base-config", default="configs/full_lora_config.yaml")
        sp.add_argument("--data-root", default="data")
        sp.add_argument("--trainer", default="train_sam3_lora_native_claude.py")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--gpus", type=int, default=1,
                        help="A1.27: GPUs per training; N>1 adds --device and "
                             "divides grad accumulation by N (make_jobs passes it)")

    d = sub.add_parser("decide")
    d.add_argument("--pilot-run", required=True)
    d.add_argument("--base", type=int, required=True)
    d.add_argument("--ext", type=int, required=True)
    common(d)
    e = sub.add_parser("extend")
    e.add_argument("--run", required=True)
    common(e)
    t = sub.add_parser("train")
    t.add_argument("--fold", required=True)
    t.add_argument("--seed", type=int, default=0)
    common(t)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "decide":
        return cmd_decide(a)
    if a.cmd == "extend":
        return cmd_extend(a)
    if a.cmd == "train":
        return cmd_train(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
