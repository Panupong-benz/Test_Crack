# -*- coding: utf-8 -*-
"""Sequential job queue for the vast.ai instance — no idle minutes.

Reads jobs.yaml (list of {name, cmd, [after]}), runs each in order, logs to
runs/<name>.log, writes queue_state.json after every job so an interrupted
instance resumes where it stopped (pair with train_seg --resume). On queue
completion touches QUEUE_DONE and optionally powers the instance off.

  python queue_runner.py --jobs jobs.yaml [--poweroff]

Rules:
- a job with "after": <name> is SKIPPED (and the queue carries on) when that
  job did not succeed. It used to `break` the whole loop, which meant one
  skipped leaf ended the run - see Amendment A1.5.
- a failed job stops the queue (money-safe default) unless "optional": true.
  This is what still enforces the kill-gate: the gate chain is not optional,
  so a bad gate returns 1 here before anything expensive follows.
- so the two flags say different things: "optional" = this failing must not
  stop the queue; "after" = do not bother running me if my input is missing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--poweroff", action="store_true")
    args = ap.parse_args()

    jobs = yaml.safe_load(args.jobs.read_text())["jobs"]
    state_f = Path("queue_state.json")
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    Path("runs").mkdir(exist_ok=True)

    for job in jobs:
        name = job["name"]
        if state.get(name) == "ok":
            print(f"[skip] {name} (done)")
            continue
        dep = job.get("after")
        if dep and state.get(dep) != "ok":
            # continue, NOT break: an optional pred/eval that failed must cost
            # us only its own dependents, never the trainings and the archive
            # that follow it in the queue (Amendment A1.5).
            print(f"[skip] {name} needs {dep} which is not ok")
            state[name] = f"skipped({dep})"
            state_f.write_text(json.dumps(state, indent=2))
            continue
        print(f"[run ] {name}: {job['cmd']}")
        t0 = time.time()
        with open(f"runs/{name}.log", "a", encoding="utf-8") as log:
            r = subprocess.run(job["cmd"], shell=True,
                               stdout=log, stderr=subprocess.STDOUT)
        state[name] = "ok" if r.returncode == 0 else f"exit{r.returncode}"
        state[f"{name}_hours"] = round((time.time() - t0) / 3600, 3)
        state_f.write_text(json.dumps(state, indent=2))
        print(f"[{state[name]:>4}] {name} ({state[f'{name}_hours']} h)")
        if r.returncode != 0 and not job.get("optional"):
            print("queue stopped on failure (money-safe default)")
            return 1

    Path("QUEUE_DONE").write_text(json.dumps(state, indent=2))
    print("queue complete")
    if args.poweroff:
        subprocess.run("poweroff", shell=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
