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
    """A6: JSONL of {epoch, train_loss, val_loss} — lower better.

    Returns (vals, meta). The trainer appends per epoch and (before A1.21)
    never rotated, so a retried fold CONCATENATED two runs' curves - measured:
    an attempt dead at epoch 9 plus a full retry turned best_epoch 12 into 21
    with the verdict unchanged. The frozen criterion says "the run's
    validation curve"; a concatenation was never that, so keeping only the
    LAST contiguous epoch block is a reading-bug fix, not a criterion change
    (A1.21 item 111). What was discarded is reported, never hidden:
    meta = {"restarts": blocks - 1, "epochs_discarded": len of earlier blocks}.
    """
    fp = run / "val_stats.json"
    if not fp.exists():
        return None, {}
    recs = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            recs.append((int(j.get("epoch", len(recs) + 1)),
                         float(j["val_loss"])))
        except (ValueError, KeyError, TypeError):
            continue
    blocks, cur, last_e = [], [], 0
    for e, v in recs:
        if e <= last_e and cur:          # epoch number reset = a new attempt
            blocks.append(cur)
            cur = []
        cur.append(v)
        last_e = e
    if cur:
        blocks.append(cur)
    vals = blocks[-1] if blocks else []
    meta = {"restarts": max(0, len(blocks) - 1),
            "epochs_discarded": sum(len(b) for b in blocks[:-1])}
    return vals, meta


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
    if rj.exists():                        # A2-A4, and A6 since A1.21 item 112
        try:
            j = json.loads(rj.read_text(encoding="utf-8"))
        except ValueError:
            j = {}
        ep, bs = j.get("epochs"), j.get("batch")
        tiles = j.get("n_train_tiles")
        # A6 accumulates gradients: (tiles // bs) counts MICRO-steps, so
        # divide by grad_accum or A6's ~12,200 optimizer steps would be
        # reported as ~97,600. A2-A4 run.json has no grad_accum key -> 1.
        acc = int(j.get("grad_accum", 1) or 1)
        rf = j.get("resumed_from") or ""
        row.update(epochs=ep or "", batch=bs or "", grad_accum=acc,
                   n_train_tiles=tiles or "", budget_source="run.json",
                   resumed_from=("/".join(str(x) for x in rf)
                                 if isinstance(rf, list) else rf))
        if ep and bs and tiles:
            row["optimizer_steps"] = (tiles // bs) * ep // acc
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
        # A6 keeps batch_size under training:, which the old lookup never
        # checked - the column was blank on every A6 row (A1.21 item 113)
        bs = (tr.get("batch_size") or dl.get("batch_size")
              or (c or {}).get("batch_size"))
        acc = tr.get("gradient_accumulation_steps") or 1
        row.update(epochs=ep or "", batch=bs or "", grad_accum=acc,
                   budget_source="config")
        return row
    if model == "a1":                                  # nnU-Net, fixed by it
        row.update(epochs=250, batch="self-config", grad_accum=1,
                   optimizer_steps=250 * 250, budget_source="a1-fixed")
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
        brow = budget_row(run, model, cfg_dir, hours.get(run.name))
        budgets.append(brow)

        meta = {}
        if model in ("unet", "deeplabv3p", "segformer"):
            vals, higher = read_valid_log(run), True
        elif model == "a6":
            vals, meta = read_val_stats(run)
            higher = False
        else:                                          # a1
            vals, higher = read_nnunet_log(nnunet_results, fold), True

        budget = brow["epochs"]
        bknown = 1 if isinstance(budget, int) and budget > 0 else 0
        if not bknown and vals:
            # A1.21 item 112: with no budget, judge() falls back to the
            # logged length, under which clause 1 passes for almost any
            # curve. The fallback stays (changing it would change the frozen
            # criterion's output) but it must never be silent, and summarize
            # --strict refuses a table built on it.
            print(f"WARN {run.name}: budget unknown (no run.json and no "
                  f"config) - verdict is judged against the logged length, "
                  f"which biases toward 'saturated'")
        r = judge(vals or [], budget if bknown else 0, higher)
        if model == "a1" and r["verdict"] != "no_log":
            r["verdict"] = "weak_evidence"             # -f all: no held-out val
        restarts = int(meta.get("restarts", 0))
        curve_ok = 1 if (restarts == 0 and not (
            bknown and r.get("epochs_logged", 0) > budget)) else 0
        rows.append({"run": run.name, "model": model, "fold": fold,
                     "seed": seed if seed is not None else "", **r,
                     "restarts": restarts,
                     "epochs_discarded": meta.get("epochs_discarded", 0),
                     "curve_ok": curve_ok,
                     "budget_source": brow.get("budget_source", ""),
                     "budget_known": bknown})

    results.mkdir(parents=True, exist_ok=True)
    fields = ["run", "model", "fold", "seed", "verdict", "epochs_logged",
              "budget_epochs", "best_epoch", "best_frac", "N", "total_gain",
              "tail_gain", "rel_tail_gain", "clause1", "clause2",
              # A1.21: transparency columns, NOT new verdicts - the verdict
              # set is frozen. restarts/epochs_discarded say what the reader
              # had to discard; curve_ok=0 flags a file that was not a single
              # clean run; budget_source/budget_known expose the budget-or-n
              # fallback instead of letting it pass as a real budget.
              "restarts", "epochs_discarded", "curve_ok",
              "budget_source", "budget_known"]
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
                                           "hours", "budget_source",
                                           "resumed_from"])
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
    # A1.21: a retried A6 fold concatenates val_stats.json. The reader must
    # keep only the last contiguous block and SAY what it discarded.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runs = root / "runs"

        def write_stats(d, curves):
            lines = []
            for c in curves:
                for i, v in enumerate(c, 1):
                    lines.append(json.dumps(
                        {"epoch": i, "train_loss": v, "val_loss": v}))
            (d / "val_stats.json").write_text("\n".join(lines) + "\n")

        # the dead attempt reached a LOWER loss (10.0) than the real run's
        # best (~40 at epoch 12): the old reader would report the ghost
        dead = [300.0 - 30 * i for i in range(9)]
        dead[-1] = 10.0
        real = [300 * 0.82 ** min(i, 12) + 40 for i in range(1, 31)]
        d1 = runs / "a6_F1_s0"
        d1.mkdir(parents=True)
        (d1 / "run.json").write_text(json.dumps(
            {"epochs": 30, "batch": 2, "grad_accum": 8,
             "n_train_tiles": 6000}))
        write_stats(d1, [dead, real])
        d2 = runs / "a6_F2_s0"                # no run.json, no config -> WARN
        d2.mkdir(parents=True)
        write_stats(d2, [real])
        res = root / "res"
        rows2 = collect(runs, res, root / "cfg", root / "queue_state.json",
                        None)
        r1 = next(r for r in rows2 if r["run"] == "a6_F1_s0")
        assert r1["restarts"] == 1 and r1["epochs_discarded"] == 9, r1
        assert r1["curve_ok"] == 0, r1
        assert r1["epochs_logged"] == 30 and r1["budget_epochs"] == 30, r1
        assert r1["best_epoch"] == 12, (
            "best must come from the LAST block, not the dead attempt's "
            "ghost minimum", r1)
        assert r1["verdict"] in ("saturated", "budget_limited",
                                 "degenerate"), r1
        assert r1["budget_source"] == "run.json" and r1["budget_known"] == 1
        r2 = next(r for r in rows2 if r["run"] == "a6_F2_s0")
        assert r2["budget_known"] == 0 and r2["budget_source"] == "", r2
        assert r2["restarts"] == 0 and r2["curve_ok"] == 1, r2
        bt2 = list(csv.DictReader(open(res / "budget_table.csv",
                                       encoding="utf-8")))
        b1 = next(b for b in bt2 if b["run"] == "a6_F1_s0")
        assert int(b1["optimizer_steps"]) == (6000 // 2) * 30 // 8, (
            "grad_accum must divide micro-steps", b1)
        assert b1["budget_source"] == "run.json", b1
    # A1.22: an extended run appends epochs 31..60 to the same file - that is
    # a CONTINUATION (one block, restarts=0), not a restart, and the budget
    # comes from the rewritten run.json (epochs=60, resumed_from=[30])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / "runs" / "a6_F1_s0"
        d.mkdir(parents=True)
        curve = [300 * 0.82 ** min(i, 12) + 40 for i in range(1, 61)]
        (d / "val_stats.json").write_text("".join(
            json.dumps({"epoch": i, "train_loss": v, "val_loss": v}) + "\n"
            for i, v in enumerate(curve, 1)))
        (d / "run.json").write_text(json.dumps(
            {"epochs": 60, "batch": 2, "grad_accum": 8, "n_train_tiles": 6000,
             "restarts": 0, "resumed_from": [30]}))
        res = root / "res"
        rows3 = collect(root / "runs", res, root / "cfg",
                        root / "queue_state.json", None)
        r = rows3[0]
        assert r["restarts"] == 0 and r["curve_ok"] == 1, r
        assert r["epochs_logged"] == 60 and r["budget_epochs"] == 60, r
        assert r["N"] == 6, r                       # tail_window(60)
        bt3 = list(csv.DictReader(open(res / "budget_table.csv",
                                       encoding="utf-8")))
        assert bt3[0]["resumed_from"] == "30", bt3[0]
    print("selftest PASS: saturated / budget_limited / creeping->saturated "
          "(OR, not AND) / degenerate; mirrored for lower-is-better; "
          "N=3,4,25 for B=30,40,250; end-to-end runs/ scan + budget maths; "
          "continued 1..60 curve = one block (restarts=0, budget 60, "
          "resumed_from surfaced); "
          "concatenated curve -> last block only (restarts=1, curve_ok=0, "
          "ghost minimum ignored); unknown budget -> budget_known=0 + WARN; "
          "a6 optimizer steps divided by grad_accum")
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
