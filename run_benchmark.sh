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
[ -f /workspace/.bm_env ] && source /workspace/.bm_env   # nnU-Net dirs

# resource monitor (Amendment A1.9): background sampler -> results/benchmark/
# resource_log.csv. Pidfile-guarded, exits on its own after QUEUE_DONE; on the
# full run poweroff kills it, which is fine - the report is a queue job.
start_monitor() {
  nohup python benchmark/resource_monitor.py --daemon       >> runs/resource_monitor.log 2>&1 &
  echo "resource monitor started (pid $!)"
}
mkdir -p runs

MODE="${1:-}"
case "$MODE" in
  smoke)
    start_monitor
    python benchmark/queue_runner.py --jobs benchmark/jobs_template.yaml
    python benchmark/resource_monitor.py --report
    ;;
  smoke-a6)
    # interim A6-only rental (Amendment A1.8): no seg-arch batch sweep -
    # A6's batch is fixed in the config, so the smoke only proves the pipe
    # and prints the s/step for the runbook hour formula (~15 min)
    start_monitor
    python benchmark/queue_runner.py --jobs benchmark/jobs_template_a6.yaml
    python benchmark/resource_monitor.py --report
    ;;
  full)
    if [ ! -f jobs.yaml ]; then
      echo "jobs.yaml missing — generate it with the smoke-hour batch size:"
      echo "  python benchmark/make_jobs.py --batch <B> --out jobs.yaml"
      exit 2
    fi
    start_monitor
    python benchmark/queue_runner.py --jobs jobs.yaml --poweroff
    ;;
  *)
    echo "usage: $0 smoke|smoke-a6|full"; exit 2 ;;
esac
