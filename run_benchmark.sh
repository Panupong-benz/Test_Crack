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

MODE="${1:-}"
case "$MODE" in
  smoke)
    python benchmark/queue_runner.py --jobs benchmark/jobs_template.yaml
    ;;
  full)
    if [ ! -f jobs.yaml ]; then
      echo "jobs.yaml missing — generate it with the smoke-hour batch size:"
      echo "  python benchmark/make_jobs.py --batch <B> --out jobs.yaml"
      exit 2
    fi
    python benchmark/queue_runner.py --jobs jobs.yaml --poweroff
    ;;
  *)
    echo "usage: $0 smoke|full"; exit 2 ;;
esac
