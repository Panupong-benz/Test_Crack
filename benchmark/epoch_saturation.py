# -*- coding: utf-8 -*-
"""Did each row get enough epochs? Answered by a criterion fixed BEFORE the run.

Why this exists
---------------
`train_seg.py` writes `valid_log.csv` every epoch and the SAM3 trainer writes
`val_stats.json` every epoch, and until now NOTHING read either: the benchmark
scored only the final masks. So a row whose validation curve was still climbing
when its epoch cap hit would report a number that is really a LOWER BOUND, and
we would never know. That is exactly the question "40 / 30 / 250 epochs - is it
enough?" and it cannot be answered by looking at a curve afterwards, because
looking afterwards is choosing.

THE CRITERION (pre-registered - this docstring is committed before the run,
per the SS8s/SS8t pattern; changing any constant later requires its own commit
saying what changed and why):

    B          = the row's epoch budget
    N          = max(3, round(0.1 * B))        formula, not a constant: B is
                                               30 / 40 / 250 across the rows,
                                               so a fixed N means different
                                               things in different rows
    total_gain = |best - first|
    tail_gain  = |best - best_at(B - N)|

    saturated  <=>  best_epoch <= 0.8 * B                      (clause 1)
                OR  tail_gain  <= EPS_REL * total_gain         (clause 2)

    EPS_REL = 0.02

Three choices in there that are deliberate:

* **OR, not AND.** Clause 1 holding means the last 20% of training produced no
  new best at all - that IS saturation, on its own. Clause 2 exists for a
  different case: a curve that creeps up by 1e-4 per epoch and technically
  peaks at the very last epoch. Under AND that curve would be branded
  budget-limited although it is flat; under OR it is correctly called
  saturated.
* **EPS_REL is a FRACTION.** A2-A4 score a hard-Dice-like quantity in [0, 1]
  while A6 scores a composite validation LOSS (weighted 200 focal / 50 dice /
  30 clDice, unbounded above). No single absolute epsilon can serve both
  scales; "the last N epochs contributed under 2% of everything the run ever
  gained" is scale-free and reads the same on either.
* **`total_gain` near zero is `degenerate`, never `saturated`.** A curve that
  never moved is not evidence of convergence, and silently calling it
  saturated would be the most flattering possible reading.

Coverage, stated rather than implied:

* A2/A3/A4 - `runs/<tag>/valid_log.csv`, column `valid_score`, higher better.
* A6      - `runs/<tag>/val_stats.json` (JSONL), `val_loss`, lower better.
* A1      - nnU-Net trains with `-f all`, i.e. NO held-out validation, and its
            best checkpoint is the last epoch by definition. Only its own
            TRAINING pseudo-dice curve exists, which is a different quantity,
            so A1 is always reported as `weak_evidence` - never as a verdict
            comparable to the other rows.
* A5      - no training at all; excluded.

Row-level aggregation: a row counts as budget-limited if ANY of its fold/seed
runs is, reported as a count ("5/12"). One rule for every row, ours included -
and it is worth recording here, before any number exists, that A6 has the
FEWEST optimizer updates of the trained rows (~12k against A2-A4's ~32k), so
A6 is a priori the likeliest row to hit its ceiling.

Outputs
  results/benchmark/epoch_saturation.csv   per run: verdict + the numbers
  results/benchmark/budget_table.csv       per run: what the budget really was
                                           (epochs, batch, steps, tiles, hours)

Usage
  python benchmark/epoch_saturation.py [--runs runs] [--results results/benchmark]
  python benchmark/epoch_saturation.py --selftest

Always exits 0 when it merely finds nothing to read: a missing curve is
reported as `no_log`, not as a failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

# ---- the pre-registered constants -----------------------------------------
BEST_FRAC_MAX = 0.8      # clause 1: best epoch within the first 80% of budget
EPS_REL = 0.02           # clause 2: tail contributed < 2% of all gain
N_FRAC = 0.1             # tail window = max(N_MIN, round(N_FRAC * budget))
N_MIN = 3
DEGENERATE_ABS = 1e-9    # total_gain at or below this = degenerate

TAG_RE = re.compile(r"^(a1|a5|a6|unet|deeplabv3p|segformer)_"
                    r"([A-Za-z0-9]+?)(?:_s(\d+))?$")
NNUNET_DICE_RE = re.compile(r"Pseudo dice\s*\[?([0-9.]+)")


def tail_window(budget: int) -> int:
    return max(N_MIN, int(round(N_FRAC * budget)))


def judge(values, budget: int, higher_is_better: bool) -> dict:
    """values: per-epoch scores in epoch order (1-based epochs)."""
    if not values:
        return {"verdict": "no_log"}
    n = len(values)
    budget = budget or n
    sign = 1.0 if higher_is_better else -1.0
    signed = [sign * v for v in values]           # now always higher = better

    best_i = max(range(n), key=lambda i: signed[i])
    best_epoch = best_i + 1
    best = signed[best_i]
    first = signed[0]
    total_gain = best - first

    N = tail_window(budget)
    cut = max(0, n - N)                            # epochs before the tail
    best_before_tail = max(signed[:cut]) if cut else first
    tail_gain = max(0.0, best - best_before_tail)

    best_frac = best_epoch / budget if budget else 1.0
    clause1 = best_frac <= BEST_FRAC_MAX
    rel = (tail_gain / total_gain) if total_gain > DEGENERATE_ABS else None
    clause2 = rel is not None and rel <= EPS_REL

    if total_gain <= DEGENERATE_ABS:
        verdict = "degenerate"
    elif clause1 or clause2:
        verdict = "saturated"
    else:
        verdict = "budget_limited"
    return {"verdict": verdict, "epochs_logged": n, "budget_epochs": budget,
            "best_epoch": best_epoch, "best_frac": round(best_frac, 4),
            "N": N, "total_gain": total_gain, "tail_gain": tail_gain,
            "rel_tail_gain": rel, "clause1": int(clause1),
            "clause2": int(clause2)}


# ------------------------------------------------------------------ readers -
def read_valid_log(run: Path):
    """A2-A4: epoch,step,valid_score,best,sec — higher better."""
    fp = run / "valid_log.csv"
    if not fp.exists():
        return None
    out = []
    with open(fp, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(float(row["valid_score"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def read_val_stats(run: Path):
    """A6: JSONL of {epoch, train_loss, val_loss} — lower better."""
    fp = run / "val_stats.json"
    if not fp.exists():
        return None
    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(float(json.loads(line)["val_loss"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def read_nnunet_log(results_root: Path | None, fold: str):
    """A1, best effort. nnU-Net prints 'Pseudo dice [0.42]' per epoch into
    training_log_<date>.txt. This is a TRAINING curve on the same data it
    fits (-f all, no held-out split), so it can never carry the same weight
    as the other rows - the caller stamps weak_evidence regardless."""
    if not results_root or not results_root.exists():
        return None
    logs = sorted(results_root.glob("**/training_log_*.txt"))
    if not logs:
        return None
    vals = []
    for line in logs[-1].read_text(encoding="utf-8",
                                   errors="ignore").splitlines():
        m = NNUNET_DICE_RE.search(line)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return vals or None


# ------------------------------------------------------------------ budgets -
def budget_row(run: Path, model: str, cfg_dir: Path, hours):
    """What this run's budget REALLY was, in numbers a reviewer can check."""
    row = {"run": run.name, "model": model, "epochs": "", "batch": "",
           "grad_accum": "", "n_train_tiles": "", "optimizer_steps": "",
           "samples_seen": "", "hours": hours if hours is not None else ""}
    rj = run / "run.json"
    if rj.exists():                                    # A2-A4
        try:
            j = json.loads(rj.read_text(encoding="utf-8"))
        except ValueError:
            j = {}
        ep, bs = j.get("epochs"), j.get("batch")
        tiles = j.get("n_train_tiles")
        row.update(epochs=ep or "", batch=bs or "", grad_accum=1,
                   n_train_tiles=tiles or "")
        if ep and bs and tiles:
            row["optimizer_steps"] = (tiles // bs) * ep
            row["samples_seen"] = tiles * ep
        return row
    cfg = cfg_dir / f"{run.name}.yaml"                 # A6
    if cfg.exists():
        try:
            import yaml
            c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            c = {}
        tr = (c or {}).get("training", {}) or {}
        dl = (c or {}).get("dataloader", {}) or (c or {}).get("data", {}) or {}
        ep = tr.get("num_epochs")
        bs = dl.get("batch_size") or (c or {}).get("batch_size")
        acc = tr.get("gradient_accumulation_steps") or 1
        row.update(epochs=ep or "", batch=bs or "", grad_accum=acc)
        return row
    if model == "a1":                                  # nnU-Net, fixed by it
        row.update(epochs=250, batch="self-config", grad_accum=1,
                   optimizer_steps=250 * 250)
    return row


# --------------------------------------------------------------------- main -
def collect(runs_dir: Path, results: Path, cfg_dir: Path,
            queue_state: Path, nnunet_results: Path | None):
    hours = {}
    if queue_state.exists():
        try:
            st = json.loads(queue_state.read_text(encoding="utf-8"))
            hours = {k[:-6]: v for k, v in st.items() if k.endswith("_hours")}
        except ValueError:
            pass

    rows, budgets = [], []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()) \
            if runs_dir.exists() else []:
        m = TAG_RE.match(run.name)
        if not m:
            continue
        model, fold, seed = m.group(1), m.group(2), m.group(3)
        if model == "a5":
            continue                                   # zero-shot, no training
        budgets.append(budget_row(run, model, cfg_dir, hours.get(run.name)))

        if model in ("unet", "deeplabv3p", "segformer"):
            vals, higher = read_valid_log(run), True
        elif model == "a6":
            vals, higher = read_val_stats(run), False
        else:                                          # a1
            vals, higher = read_nnunet_log(nnunet_results, fold), True

        budget = next((b["epochs"] for b in budgets if b["run"] == run.name),
                      None)
        r = judge(vals or [], budget if isinstance(budget, int) else 0, higher)
        if model == "a1" and r["verdict"] != "no_log":
            r["verdict"] = "weak_evidence"             # -f all: no held-out val
        rows.append({"run": run.name, "model": model, "fold": fold,
                     "seed": seed if seed is not None else "", **r})

    results.mkdir(parents=True, exist_ok=True)
    fields = ["run", "model", "fold", "seed", "verdict", "epochs_logged",
              "budget_epochs", "best_epoch", "best_frac", "N", "total_gain",
              "tail_gain", "rel_tail_gain", "clause1", "clause2"]
    with open(results / "epoch_saturation.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(results / "budget_table.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "model", "epochs", "batch",
                                           "grad_accum", "n_train_tiles",
                                           "optimizer_steps", "samples_seen",
                                           "hours"])
        w.writeheader()
        w.writerows(budgets)
    return rows


def row_summary(rows):
    """{model: 'k/n'} for the rows that hit their ceiling. Any run counts."""
    out = {}
    for r in rows:
        n, k = out.get(r["model"], (0, 0))
        out[r["model"]] = (n + 1, k + (r["verdict"] == "budget_limited"))
    return {m: f"{k}/{n}" for m, (n, k) in out.items() if k}


def selftest():
    """Four planted curves. The creeping one is the point: under the AND form
    of this criterion it would come back budget_limited, which is why the
    rule is OR."""
    import tempfile
    B = 40

    def line(vals):
        return judge(vals, B, higher_is_better=True)["verdict"]

    # 1. clean saturation: peaks at epoch 20 of 40, flat after
    clean = [0.1 + 0.03 * i for i in range(20)] + [0.67] * 20
    # 2. still climbing hard at the cap
    climb = [0.1 + 0.015 * i for i in range(B)]
    # 3. creeps to the last epoch by a hair after an early rise
    creep = [0.1 + 0.03 * i for i in range(20)] + \
            [0.67 + 1e-4 * i for i in range(20)]
    # 4. never moved
    flat = [0.42] * B
    assert line(clean) == "saturated", line(clean)
    assert line(climb) == "budget_limited", line(climb)
    assert line(creep) == "saturated", (
        "the creeping curve must read as saturated - if this says "
        "budget_limited the criterion has been changed to AND")
    assert line(flat) == "degenerate", line(flat)

    # lower-is-better must mirror it exactly
    assert judge([-v for v in climb], B, higher_is_better=False)["verdict"] \
        == "budget_limited"
    assert judge([-v for v in clean], B,
                 higher_is_better=False)["verdict"] == "saturated"

    # N is a formula, not a constant
    assert tail_window(30) == 3 and tail_window(40) == 4 \
        and tail_window(250) == 25

    # end to end over a fake runs/ tree
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runs = root / "runs"
        for tag, vals in (("unet_F1_s0", clean), ("unet_F1_s1", climb)):
            d = runs / tag
            d.mkdir(parents=True)
            (d / "run.json").write_text(json.dumps(
                {"epochs": B, "batch": 8, "n_train_tiles": 800}))
            with open(d / "valid_log.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["epoch", "step", "valid_score", "best", "sec"])
                for i, v in enumerate(vals, 1):
                    w.writerow([i, i * 100, v, v, i])
        (runs / "a5_F1").mkdir(parents=True)           # must be ignored
        res = root / "res"
        rows = collect(runs, res, root / "cfg", root / "queue_state.json",
                       None)
        got = {r["run"]: r["verdict"] for r in rows}
        assert got == {"unet_F1_s0": "saturated",
                       "unet_F1_s1": "budget_limited"}, got
        assert row_summary(rows) == {"unet": "1/2"}, row_summary(rows)
        bt = list(csv.DictReader(open(res / "budget_table.csv",
                                      encoding="utf-8")))
        b0 = next(b for b in bt if b["run"] == "unet_F1_s0")
        assert int(b0["optimizer_steps"]) == (800 // 8) * B, b0
        assert int(b0["samples_seen"]) == 800 * B, b0
    print("selftest PASS: saturated / budget_limited / creeping->saturated "
          "(OR, not AND) / degenerate; mirrored for lower-is-better; "
          "N=3,4,25 for B=30,40,250; end-to-end runs/ scan + budget maths")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results/benchmark"))
    ap.add_argument("--config-dir", type=Path,
                    default=Path("configs/benchmark"))
    ap.add_argument("--queue-state", type=Path,
                    default=Path("queue_state.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    nn = os.environ.get("nnUNet_results")
    rows = collect(a.runs, a.results, a.config_dir, a.queue_state,
                   Path(nn) if nn else None)
    if not rows:
        print(f"no trained runs under {a.runs} - nothing to judge "
              f"(not a failure)")
        return 0
    for r in rows:
        print(f"  {r['run']:<24} {r['verdict']:<15} "
              f"best {r.get('best_epoch', '?')}/{r.get('budget_epochs', '?')}"
              f"  rel_tail {r.get('rel_tail_gain')}")
    lim = row_summary(rows)
    print(f"\nbudget-limited rows: {lim or 'none'}")
    print(f"-> {a.results}/epoch_saturation.csv, {a.results}/budget_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
