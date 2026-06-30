#!/usr/bin/env bash
# run_train.sh — operational wrapper around run_lowo.py for vast.ai.
#
# It does NOT reimplement the sweep (run_lowo.py already does that). It adds the
# layer run_lowo.py lacks: pre-flight checks, an off-box checkpoint backup loop
# (so an instance death doesn't lose finished folds), timestamped logging, and a
# simple fold argument.
#
# RUN IT INSIDE tmux so a dropped connection can't kill the job:
#     tmux new -s train
#     ./run_train.sh RW20      # one fold (progress report)
#     ./run_train.sh all       # both folds (full LOWO)
#   then press Ctrl-b then d to detach. Reattach later: tmux attach -t train
#   (no tmux? use:  nohup ./run_train.sh all > nohup.out 2>&1 & )
#
# Resume reality: the trainer has no mid-fold resume. With SKIP_EXISTING=1,
# re-running skips folds that already produced best_lora_weights.pt, but a fold
# interrupted mid-way restarts from scratch. The backup loop protects FINISHED
# folds' weights by copying them off the box.

set -euo pipefail

# ===================== CONFIG (edit) =====================
REPO_DIR="/Test_Crack"                              # holds run_lowo.py + the trainer
LOWO_ROOT="/workspace/folds"                        # where fold_*/ live (after unzip)
GDRIVE_ID=""                                        # Google Drive FILE ID *or* full share URL of the folds zip
                                                    # empty = no download, use LOWO_ROOT as-is
DATA_DIR="/workspace"                               # download + unzip target
ZIP_NAME="folds.zip"
BASE_CONFIG="${REPO_DIR}/configs/full_lora_config.yaml"
RUNS_DIR="/workspace/outputs/lowo"                  # where weights/logs land
BACKUP_DEST=""                                      # rsync target off the box, e.g.
                                                    #   user@1.2.3.4:/home/me/thesis_runs
                                                    #   or a mounted persistent volume path
                                                    # empty = no off-box backup (NOT recommended)
BACKUP_EVERY=600                                    # seconds between off-box backups
EPOCHS=""                                           # override epochs (empty = use config)
DEVICE=""                                           # e.g. "0" or "0 1"; empty = trainer default
SKIP_EXISTING=1                                     # 1 = skip folds already finished (resume-ish)
KEEP_GOING=1                                        # 1 = one fold failing won't abort the others
# ========================================================

FOLD_ARG="${1:-}"
if [[ -z "${FOLD_ARG}" ]]; then
  echo "usage: $0 <RW20|RW20T|all>"; exit 2
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*"; }

# ---------- pre-flight ----------
say "pre-flight checks"
cd "${REPO_DIR}" || { echo "❌ REPO_DIR not found: ${REPO_DIR}"; exit 1; }
[[ -f run_lowo.py ]]          || { echo "❌ run_lowo.py not in ${REPO_DIR}"; exit 1; }
[[ -f "${BASE_CONFIG}" ]]     || { echo "❌ base config not found: ${BASE_CONFIG}"; exit 1; }

# ---------- fetch folds from Google Drive (optional) ----------
if [[ -n "${GDRIVE_ID}" ]]; then
  have_folds=""
  for cand in "${LOWO_ROOT}" "${DATA_DIR}" "${DATA_DIR}/folds"; do
    if compgen -G "${cand}/fold_*" >/dev/null 2>&1; then have_folds="${cand}"; break; fi
  done
  if [[ -n "${have_folds}" ]]; then
    say "folds already present at ${have_folds}; skipping download"
  else
    command -v gdown >/dev/null 2>&1 || { say "installing gdown"; python3 -m pip install -q gdown || { echo "❌ gdown install failed"; exit 1; }; }
    command -v unzip >/dev/null 2>&1 || { say "installing unzip"; apt-get update -y >/dev/null 2>&1 && apt-get install -y unzip >/dev/null 2>&1 || true; }
    mkdir -p "${DATA_DIR}"
    say "downloading folds zip from Google Drive -> ${DATA_DIR}/${ZIP_NAME}"
    if [[ "${GDRIVE_ID}" == *"/"* || "${GDRIVE_ID}" == http* ]]; then
      gdown --fuzzy "${GDRIVE_ID}" -O "${DATA_DIR}/${ZIP_NAME}" || { echo "❌ gdown failed (big-file quota? try the full share URL, or retry later)"; exit 1; }
    else
      gdown "${GDRIVE_ID}" -O "${DATA_DIR}/${ZIP_NAME}" || { echo "❌ gdown failed (big-file quota? try the full share URL, or retry later)"; exit 1; }
    fi
    say "unzipping -> ${DATA_DIR}"
    unzip -o -q "${DATA_DIR}/${ZIP_NAME}" -d "${DATA_DIR}" || { echo "❌ unzip failed"; exit 1; }
  fi
  # the zip may unpack with or without a wrapper folder; point LOWO_ROOT at the real fold_*/ location
  for cand in "${LOWO_ROOT}" "${DATA_DIR}" "${DATA_DIR}/folds"; do
    if compgen -G "${cand}/fold_*" >/dev/null 2>&1; then LOWO_ROOT="${cand}"; break; fi
  done
  say "using LOWO_ROOT=${LOWO_ROOT}"
fi

[[ -d "${LOWO_ROOT}" ]]       || { echo "❌ LOWO_ROOT not found: ${LOWO_ROOT}"; exit 1; }

# at least one fold with a real train annotation
shopt -s nullglob
folds=( "${LOWO_ROOT}"/fold_*/ )
shopt -u nullglob
[[ ${#folds[@]} -gt 0 ]] || { echo "❌ no fold_*/ under ${LOWO_ROOT} — run lowo_split.py"; exit 1; }
for f in "${folds[@]}"; do
  [[ -f "${f}train/_annotations.coco.json" ]] || { echo "❌ ${f} missing train/_annotations.coco.json"; exit 1; }
done
say "found ${#folds[@]} fold(s): $(printf '%s ' "${folds[@]##*/fold_}" | sed 's#/##g')"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
else
  say "⚠️ nvidia-smi not found — is this a GPU box? continuing anyway"
fi
[[ -n "${HF_TOKEN:-}" ]] || say "⚠️ HF_TOKEN not set in env — trainer may fail to fetch SAM3 weights"

mkdir -p logs "${RUNS_DIR}"
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
say "logging to ${REPO_DIR}/${LOG}"

# ---------- off-box backup loop ----------
BK_PID=""
backup_once() {
  [[ -z "${BACKUP_DEST}" ]] && return 0
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial "${RUNS_DIR}/" "${BACKUP_DEST}/" 2>>"${LOG}" || say "⚠️ rsync backup failed (see ${LOG})"
  elif [[ "${BACKUP_DEST}" != *:* ]]; then                    # local dest, no rsync -> cp
    mkdir -p "${BACKUP_DEST}" && cp -a "${RUNS_DIR}/." "${BACKUP_DEST}/" 2>>"${LOG}" || say "⚠️ cp backup failed"
  else
    say "⚠️ rsync missing and BACKUP_DEST is remote — backup SKIPPED. Install rsync on the instance."
  fi
}
if [[ -n "${BACKUP_DEST}" ]]; then
  if ! command -v rsync >/dev/null 2>&1 && [[ "${BACKUP_DEST}" == *:* ]]; then
    say "⚠️ BACKUP_DEST is remote but rsync missing — backups will skip (apt-get install -y rsync)"
  fi
  say "off-box backup every ${BACKUP_EVERY}s -> ${BACKUP_DEST}"
  ( while true; do sleep "${BACKUP_EVERY}"; backup_once; done ) &
  BK_PID=$!
else
  say "⚠️ BACKUP_DEST empty — no off-box backup. If the instance dies, finished weights are lost."
fi
cleanup() {
  if [[ -n "${BK_PID}" ]]; then
    pkill -P "${BK_PID}" 2>/dev/null || true   # kill the sleep child first
    kill "${BK_PID}" 2>/dev/null || true       # then the loop subshell
  fi
}
trap cleanup EXIT INT TERM

# ---------- build run_lowo.py command ----------
cmd=( python3 run_lowo.py
      --lowo-root  "${LOWO_ROOT}"
      --base-config "${BASE_CONFIG}"
      --runs-dir   "${RUNS_DIR}" )
[[ "${FOLD_ARG}" != "all" ]] && cmd+=( --folds "${FOLD_ARG}" )
[[ -n "${EPOCHS}" ]]         && cmd+=( --epochs "${EPOCHS}" )
[[ -n "${DEVICE}" ]]         && cmd+=( --device ${DEVICE} )
[[ "${SKIP_EXISTING}" == "1" ]] && cmd+=( --skip-existing )
[[ "${KEEP_GOING}" == "1" ]]    && cmd+=( --keep-going )

say "launching: ${cmd[*]}"
echo "============================================================"
set +e
"${cmd[@]}" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e
echo "============================================================"

# ---------- finish ----------
say "training process exited with code ${rc}; final backup"
backup_once
if [[ ${rc} -eq 0 ]]; then
  say "✅ DONE. summary: ${RUNS_DIR}/lowo_runs_summary.csv"
  ls "${RUNS_DIR}"/fold_*/best_lora_weights.pt 2>/dev/null | sed 's/^/    weights: /' || true
else
  say "❌ run_lowo.py returned ${rc} — inspect ${LOG}"
fi
exit ${rc}
