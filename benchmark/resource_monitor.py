# -*- coding: utf-8 -*-
"""Record what the rental actually consumes - disk, VRAM, RAM - so the next
instance is sized from measurement, not guesswork.

Why this exists
---------------
The runbook says "disk >= 60 GB"; the user reports the previous round peaked
around ~45 GB. Nothing on the instance ever recorded that number - it was read
off a dashboard and remembered. This monitor samples the machine during the
queue and leaves two artifacts in `results/benchmark/` (which
`collect_results.py` already tars):

  resource_log.csv        one row per sample
  resource_summary.json   peaks + the number that answers "how many GB next
                          time": MIN disk free, PEAK disk used, peak VRAM/RAM,
                          and a per-job table via queue_state.json's _running

Same reasoning as budget_table.csv and timing.csv: a claim about resources
must be checkable, and the only moment it can be measured is while the box is
alive.

Design constraints it works around:
* `run_benchmark.sh full` -> `queue_runner --poweroff` powers off before
  control returns to bash, so the REPORT is a queue job (before `collect`),
  never a trailing shell command.
* Every reader is best-effort: no nvidia-smi (or a Windows dev box with no
  /proc) leaves fields empty rather than crashing - the monitor must never be
  the thing that breaks a run. The selftest passes on machines with neither.
* `du` over runs/ + HF cache costs seconds, so directory sizes are sampled
  every --du-every ticks (default 10), not every tick.
* The daemon stops itself after seeing QUEUE_DONE on two consecutive ticks,
  so smoke runs (no poweroff) do not leave a stray process.

Usage
  python benchmark/resource_monitor.py --daemon [--interval 60]   # background
  python benchmark/resource_monitor.py --snapshot                 # one row
  python benchmark/resource_monitor.py --report                   # summarize
  python benchmark/resource_monitor.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GB = 1024 ** 3
FIELDS = ["ts", "disk_used_gb", "disk_free_gb",
          "vram_used_mb", "vram_total_mb", "gpu_util_pct", "power_w",
          "ram_used_gb", "ram_total_gb", "running_job",
          "runs_gb", "hf_cache_gb", "pool_gb", "n_gpu"]


# ----------------------------------------------------------------- readers -
def read_disk(path: Path):
    try:
        u = shutil.disk_usage(path)
        return round(u.used / GB, 2), round(u.free / GB, 2)
    except OSError:
        return "", ""


def read_gpu():
    """(vram_used_mb, vram_total_mb, util_pct, power_w) - blanks if no
    nvidia-smi. One combined query keeps it to a single subprocess."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return aggregate_gpu_lines(r.stdout)
    except Exception:                                        # noqa: BLE001
        return "", "", "", "", ""


def aggregate_gpu_lines(text: str):
    """(vram_used_max, vram_total_of_that_gpu, util_mean, power_sum, n_gpu).
    A1.27 item 152(f): the old reader took line 0 = GPU 0 only, so on a 2x
    box the second card was invisible. VRAM is reported as the busiest
    card (that is the OOM question), util as the mean, power as the sum."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        return "", "", "", "", ""
    used, total, util, power = max(rows, key=lambda r: r[0])[:2] + (0, 0)
    util = sum(r[2] for r in rows) / len(rows)
    power = sum(r[3] for r in rows)
    fmt = lambda v: (str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip("."))
    return fmt(used), fmt(total), fmt(round(util, 2)), fmt(round(power, 2)), str(len(rows))


def read_ram():
    """(used_gb, total_gb) from /proc/meminfo; blanks off-Linux."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])              # kB
        total = info["MemTotal"] / 1024 / 1024
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024 / 1024
        return round(total - avail, 2), round(total, 2)
    except Exception:                                        # noqa: BLE001
        return "", ""


def read_running(queue_state: Path):
    try:
        return json.loads(queue_state.read_text()).get("_running", "")
    except Exception:                                        # noqa: BLE001
        return ""


def du_gb(path: Path):
    """Directory size in GB; blank when absent. Python walk, not `du`, so it
    behaves the same on the dev box and the instance."""
    if not path or not path.exists():
        return ""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return round(total / GB, 2)


def sample(watch: Path, queue_state: Path, with_du: bool, du_paths: dict):
    used, free = read_disk(watch)
    vram_u, vram_t, util, power, n_gpu = read_gpu()
    ram_u, ram_t = read_ram()
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "disk_used_gb": used, "disk_free_gb": free,
           "vram_used_mb": vram_u, "vram_total_mb": vram_t,
           "gpu_util_pct": util, "power_w": power,
           "ram_used_gb": ram_u, "ram_total_gb": ram_t,
           "running_job": read_running(queue_state),
           "runs_gb": "", "hf_cache_gb": "", "pool_gb": "", "n_gpu": n_gpu}
    if with_du:
        row["runs_gb"] = du_gb(du_paths.get("runs"))
        row["hf_cache_gb"] = du_gb(du_paths.get("hf"))
        row["pool_gb"] = du_gb(du_paths.get("pool"))
    return row


def append_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ daemon -
def daemon(args):
    pid_f = args.out_csv.parent / "resource_monitor.pid"
    if pid_f.exists():
        old = pid_f.read_text().strip()
        alive = Path(f"/proc/{old}").exists() if Path("/proc").exists() else False
        if alive:
            print(f"monitor already running (pid {old}) - not starting twice")
            return 0
    pid_f.parent.mkdir(parents=True, exist_ok=True)
    pid_f.write_text(str(os.getpid()))

    du_paths = {"runs": Path("runs"),
                "hf": Path(os.environ.get(
                    "HF_HOME", Path.home() / ".cache" / "huggingface")),
                "pool": Path(args.watch_path)}
    done_marker = Path("QUEUE_DONE")
    done_ticks, n = 0, 0
    print(f"resource monitor: every {args.interval}s -> {args.out_csv}")
    try:
        while True:
            n += 1
            append_row(args.out_csv, sample(
                Path(args.watch_path), args.queue_state,
                with_du=(n % args.du_every == 1), du_paths=du_paths))
            if args.max_samples and n >= args.max_samples:
                break
            done_ticks = done_ticks + 1 if done_marker.exists() else 0
            if done_ticks >= 2:      # queue finished and no poweroff came
                print("QUEUE_DONE seen twice - monitor exiting")
                break
            time.sleep(args.interval)
    finally:
        try:
            pid_f.unlink()
        except OSError:
            pass
    return 0


# ------------------------------------------------------------------ report -
def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: list) -> dict:
    def col(k):
        return [x for x in (_fnum(r.get(k)) for r in rows) if x is not None]

    disk_used, disk_free = col("disk_used_gb"), col("disk_free_gb")
    out = {
        "n_samples": len(rows),
        "first_ts": rows[0]["ts"] if rows else None,
        "last_ts": rows[-1]["ts"] if rows else None,
        "peak_disk_used_gb": max(disk_used) if disk_used else None,
        "final_disk_used_gb": disk_used[-1] if disk_used else None,
        "min_disk_free_gb": min(disk_free) if disk_free else None,
        "peak_vram_used_mb": max(col("vram_used_mb"), default=None),
        "peak_ram_used_gb": max(col("ram_used_gb"), default=None),
        "peak_gpu_util_pct": max(col("gpu_util_pct"), default=None),
        "peak_runs_gb": max(col("runs_gb"), default=None),
        "peak_hf_cache_gb": max(col("hf_cache_gb"), default=None),
    }
    per_job = {}
    for r in rows:
        job = r.get("running_job") or ""
        if not job:
            continue
        rec = per_job.setdefault(job, {"samples": 0, "max_vram_used_mb": None,
                                       "last_disk_used_gb": None})
        rec["samples"] += 1
        v = _fnum(r.get("vram_used_mb"))
        if v is not None:
            rec["max_vram_used_mb"] = max(rec["max_vram_used_mb"] or 0, v)
        d = _fnum(r.get("disk_used_gb"))
        if d is not None:
            rec["last_disk_used_gb"] = d
    out["per_job"] = per_job
    return out


def report(args):
    if not args.out_csv.exists():
        print(f"no {args.out_csv} - monitor never ran (not a failure)")
        return 0
    rows = list(csv.DictReader(open(args.out_csv, encoding="utf-8")))
    s = summarize(rows)
    out_json = args.out_csv.parent / "resource_summary.json"
    out_json.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"resource summary ({s['n_samples']} samples, "
          f"{s['first_ts']} .. {s['last_ts']}):")
    print(f"  disk used  peak {s['peak_disk_used_gb']} GB   "
          f"final {s['final_disk_used_gb']} GB")
    print(f"  disk free  MIN  {s['min_disk_free_gb']} GB   "
          f"<- size the next container from this")
    print(f"  VRAM peak  {s['peak_vram_used_mb']} MB   "
          f"RAM peak {s['peak_ram_used_gb']} GB   "
          f"GPU util peak {s['peak_gpu_util_pct']}%")
    for job, rec in sorted(s["per_job"].items()):
        print(f"    {job:<24} vram_max {rec['max_vram_used_mb']} MB  "
              f"disk_at_end {rec['last_disk_used_gb']} GB "
              f"({rec['samples']} samples)")
    print(f"-> {out_json}")
    return 0


# ---------------------------------------------------------------- selftest -
def selftest():
    import tempfile
    # 0. A1.27: two nvidia-smi lines -> busiest card VRAM, mean util, sum W
    two = "15873, 32607, 98, 410.5" + chr(10) + "15100, 32607, 90, 380.5" + chr(10)
    u, t, ut, pw, n = aggregate_gpu_lines(two)
    assert (u, t, n) == ("15873", "32607", "2"), (u, t, n)
    assert ut == "94" and pw == "791", (ut, pw)
    one = aggregate_gpu_lines("15873, 32607, 98, 410.5")
    assert one == ("15873", "32607", "98", "410.5", "1"), one
    assert aggregate_gpu_lines("") == ("", "", "", "", ""), "no nvidia-smi -> blanks"
    print("  aggregate_gpu_lines: 2 cards -> max VRAM / mean util / sum power; 1 card unchanged")
    # 1. report math on planted rows, including blank GPU cells
    rows = [
        {"ts": "t1", "disk_used_gb": "20.0", "disk_free_gb": "40.0",
         "vram_used_mb": "", "ram_used_gb": "8.0", "gpu_util_pct": "",
         "running_job": "", "runs_gb": "0.1", "hf_cache_gb": "3.5"},
        {"ts": "t2", "disk_used_gb": "31.5", "disk_free_gb": "28.5",
         "vram_used_mb": "21000", "ram_used_gb": "12.5",
         "gpu_util_pct": "97", "running_job": "a6_RW20_s0",
         "runs_gb": "", "hf_cache_gb": ""},
        {"ts": "t3", "disk_used_gb": "30.0", "disk_free_gb": "30.0",
         "vram_used_mb": "18000", "ram_used_gb": "11.0",
         "gpu_util_pct": "88", "running_job": "a6_RW20_s0",
         "runs_gb": "", "hf_cache_gb": ""},
    ]
    s = summarize(rows)
    assert s["peak_disk_used_gb"] == 31.5, s
    assert s["min_disk_free_gb"] == 28.5, s
    assert s["final_disk_used_gb"] == 30.0, s
    assert s["peak_vram_used_mb"] == 21000.0, s
    assert s["peak_ram_used_gb"] == 12.5, s
    assert s["peak_hf_cache_gb"] == 3.5, s
    j = s["per_job"]["a6_RW20_s0"]
    assert j["samples"] == 2 and j["max_vram_used_mb"] == 21000.0, j
    assert j["last_disk_used_gb"] == 30.0, j

    # 2. a real daemon pass on THIS machine: two samples into a temp dir.
    #    Must succeed with GPU/RAM fields possibly blank (Windows dev box).
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "resource_log.csv"
        ns = argparse.Namespace(out_csv=out, watch_path=td,
                                queue_state=Path(td) / "queue_state.json",
                                interval=0.1, du_every=10, max_samples=2)
        daemon(ns)
        got = list(csv.DictReader(open(out, encoding="utf-8")))
        assert len(got) == 2, len(got)
        assert set(got[0].keys()) == set(FIELDS), got[0].keys()
        assert _fnum(got[0]["disk_used_gb"]) is not None, got[0]
        assert _fnum(got[0]["runs_gb"]) is not None or got[0]["runs_gb"] == "", got[0]
        # report over the real rows must not crash and must write the json
        ns2 = argparse.Namespace(out_csv=out)
        report(ns2)
        assert (Path(td) / "resource_summary.json").exists()
    print("selftest PASS: peak/min/per-job maths by hand; live 2-sample "
          "daemon on this machine (blank GPU/RAM cells tolerated); report "
          "writes resource_summary.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--du-every", type=int, default=10,
                    help="sample directory sizes every N ticks (du is slow)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="stop after N samples (0 = run until QUEUE_DONE/kill)")
    ap.add_argument("--watch-path", default=os.environ.get(
        "BM_WORKSPACE", "/workspace"))
    ap.add_argument("--queue-state", type=Path,
                    default=Path("queue_state.json"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/benchmark/resource_log.csv"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.report:
        return report(args)
    if args.snapshot:
        row = sample(Path(args.watch_path), args.queue_state, with_du=True,
                     du_paths={"runs": Path("runs"),
                               "hf": Path(os.environ.get(
                                   "HF_HOME",
                                   Path.home() / ".cache" / "huggingface")),
                               "pool": Path(args.watch_path)})
        append_row(args.out_csv, row)
        print(json.dumps(row, indent=2))
        return 0
    if args.daemon:
        return daemon(args)
    print("pass one of --daemon / --snapshot / --report / --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
