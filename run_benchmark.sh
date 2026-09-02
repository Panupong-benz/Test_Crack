#!/usr/bin/env bash
# run_benchmark.sh — tmux-safe queue wrapper for the benchmark grid.
#   bash run_benchmark.sh smoke   # Phase 4a ladder (benchmark/jobs_template.yaml)
#   bash run_benchmark.sh full    # generated jobs.yaml; auto-poweroff at the end
#
# RUN INSIDE tmux (a dropped ssh must not kill the queue):
#   tmux new -s bm && bash run_benchmark.sh full
#   Ctrl-b d to detach; tmux attach -t bm to return.
# queue_runner writes queue_state.json after every job — an interrupted
# instance resumes with the exact same command (finished jobs are skipped;
# train_seg jobs additionally --resume from last.pt).
set -uo pipefail
cd "$(dirname "$0")"
[ -f /workspace/.bm_env ] && source /workspace/.bm_env   # nnU-Net dirs + PYBIN/PATH (A1.13)
# Instance count per tile varies 10x across the pool, so the allocator
# sees wildly different block sizes and fragments. A1.19: 1.30 GB was
# reserved-but-unallocated at the OOM. Caller can override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# A1.27: BM_GPUS=N -> BM_DEVICES="0 .. N-1" for the smoke scripts (smoke_test.sh,
# smoke_resume.py read it). The real queue bakes --device in via make_jobs
# --gpus, so this only shapes the smoke. Default 1 = every previous rental.
BM_GPUS="${BM_GPUS:-1}"
export BM_DEVICES="${BM_DEVICES:-$(seq -s " " 0 $((BM_GPUS - 1)))}"
echo "GPUs for smoke: ${BM_DEVICES}  (BM_GPUS=${BM_GPUS})"

# Fail HERE, not at the kill-gate. The first queue job is a multi-hour
# training run, so "No module named 'torch'" surfacing from inside it reads
# like a code bug; it is almost always a fresh shell that never activated the
# image venv. One import, before anything is launched.
if ! python3 -c "import torch" >/dev/null 2>&1; then
  echo "FATAL: the python3 on PATH cannot import torch - refusing to start."
  echo "  python3   : $(command -v python3 || echo NONE)"
  echo "  BM_PYBIN  : ${BM_PYBIN:-unset (setup_benchmark.sh has not run here)}"
  echo "  fix       : source /workspace/.bm_env   (or re-run setup_benchmark.sh)"
  exit 1
fi

# resource monitor (Amendment A1.9): background sampler -> results/benchmark/
# resource_log.csv. Pidfile-guarded, exits on its own after QUEUE_DONE; on the
# full run poweroff kills it, which is fine - the report is a queue job.
start_monitor() {
  # A1.30 item 166(d): 60 s was too coarse to see a minutes-long stall;
  # at 15 s a 3-minute dense-tile block leaves ~12 samples, enough to
  # read util+power+swap and tell H1 (data wait) from H2 (RAM thrash).
  nohup python3 benchmark/resource_monitor.py --daemon --interval 15       >> runs/resource_monitor.log 2>&1 &
  local pid=$!
  # $! is set even when the command does not exist, so the old message
  # reported a dead process as started - which is exactly what happened
  # on the image without a `python` alias (A1.13). Confirm it is alive.
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "resource monitor started (pid $pid)"
    return
  fi
  # The launcher also exits 0 when a daemon is ALREADY running (pidfile
  # guard), so a dead $pid is not proof of failure - it is the benign case
  # after smoke-a6. Distinguish them before crying wolf (Amendment A1.18).
  local old_pid=""
  [ -f results/benchmark/resource_monitor.pid ] &&       old_pid="$(cat results/benchmark/resource_monitor.pid 2>/dev/null)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "resource monitor already running (pid $old_pid) - reusing it"
  else
    echo "WARN: resource monitor did NOT start - see runs/resource_monitor.log"
    tail -n 3 runs/resource_monitor.log 2>/dev/null
  fi
}
mkdir -p runs

MODE="${1:-}"
case "$MODE" in
  smoke)
    start_monitor
    python3 benchmark/queue_runner.py --jobs benchmark/jobs_template.yaml
    python3 benchmark/resource_monitor.py --report
    ;;
  smoke-a6)
    # interim A6-only rental (Amendment A1.8): no seg-arch batch sweep -
    # A6's batch is fixed in the config, so the smoke only proves the pipe
    # and prints the s/step for the runbook hour formula (~15 min)
    start_monitor
    python3 benchmark/queue_runner.py --jobs benchmark/jobs_template_a6.yaml
    python3 benchmark/resource_monitor.py --report
    ;;
  full)
    if [ ! -f jobs.yaml ]; then
      echo "jobs.yaml missing — generate it with the smoke-hour batch size:"
      echo "  python3 benchmark/make_jobs.py --batch <B> --out jobs.yaml"
      exit 2
    fi
    start_monitor
    python3 benchmark/queue_runner.py --jobs jobs.yaml --poweroff
    ;;
  *)
    echo "usage: $0 smoke|smoke-a6|full"; exit 2 ;;
esac
