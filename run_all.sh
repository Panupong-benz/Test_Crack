#!/usr/bin/env bash
# run_all.sh — unattended overnight pipeline. Chains:
#   smoke (gate) -> train each fold -> validate -> infer subset -> compare -> backup
# Trains the trainer DIRECTLY per fold (run_lowo.py is no longer in the repo), so
# there is no hidden dependency. Run it inside tmux and go to sleep.
#
#   export HF_TOKEN="hf_..."
#   tmux new -s night
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   ./run_all.sh
#   # Ctrl-b d to detach; reattach: tmux attach -t night
#
# Smoke is a GATE: if it fails, training never starts (so a bug can't burn hours).
# Off-box backup after every fold means an instance death loses at most the fold
# in progress. Set BACKUP_DEST or you are gambling the whole night.

set -uo pipefail

# ===================== CONFIG (edit) =====================
REPO_DIR="/workspace/Test_Crack"
LOWO_ROOT="/workspace/folds"                 # holds fold_RW20/ fold_RW20T/
FULL_CONFIG="${REPO_DIR}/configs/full_lora_config.yaml"
RUNS_DIR="/workspace/outputs/lowo"           # weights/logs per fold land here
RESULTS_DIR="/workspace/results"             # final metrics + compare panels collected here
FOLDS="RW20 RW20T"                           # folds to run, in order (e.g. just "RW20")
DEVICE=0

DO_SMOKE=1
DO_TRAIN=1
DO_VALIDATE=1
DO_VISUALS=1                                 # infer_subset + compare_test per fold

META_CSV="/workspace/coco_with_meta.csv"     # needed for view filtering in infer_subset
N_IMAGES=50
WANT_VIEWS="overview close-up"
THRESHOLD=0.3

BACKUP_DEST=""                               # rsync/cp target off the box (STRONGLY recommended)
AUTO_STOP=0                                  # 1 = destroy this instance after success+backup
INSTANCE_ID="${INSTANCE_ID:-}"              # needed only if AUTO_STOP=1 (vastai instance id)
# ========================================================

ts() { date +"%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*"; }
mkdir -p "${RESULTS_DIR}" "${RUNS_DIR}"
MASTER="${RESULTS_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MASTER}") 2>&1
say "pipeline start | folds: ${FOLDS} | log: ${MASTER}"

backup_ok=0
backup_now() {
  [[ -z "${BACKUP_DEST}" ]] && return 0
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial "${RESULTS_DIR}/" "${BACKUP_DEST}/results/" 2>/dev/null && \
    rsync -a --partial "${RUNS_DIR}/"    "${BACKUP_DEST}/weights/" 2>/dev/null && backup_ok=1
  elif [[ "${BACKUP_DEST}" != *:* ]]; then
    mkdir -p "${BACKUP_DEST}" && cp -a "${RESULTS_DIR}/." "${BACKUP_DEST}/results/" 2>/dev/null && \
    cp -a "${RUNS_DIR}/." "${BACKUP_DEST}/weights/" 2>/dev/null && backup_ok=1
  else
    say "⚠️ rsync missing + remote dest — backup skipped (apt-get install -y rsync)"
  fi
  [[ ${backup_ok} -eq 1 ]] && say "backed up -> ${BACKUP_DEST}"
}

# ---------- readiness ----------
cd "${REPO_DIR}" || { say "❌ REPO_DIR not found: ${REPO_DIR}"; exit 1; }
[[ -f train_sam3_lora_native_claude.py ]] || { say "❌ trainer not in repo"; exit 1; }
[[ -f "${FULL_CONFIG}" ]]                  || { say "❌ config missing: ${FULL_CONFIG}"; exit 1; }
for W in ${FOLDS}; do
  [[ -f "${LOWO_ROOT}/fold_${W}/train/_annotations.coco.json" ]] || { say "❌ fold_${W} not ready under ${LOWO_ROOT}"; exit 1; }
done
[[ -n "${HF_TOKEN:-}" ]] || say "⚠️ HF_TOKEN not set — make sure huggingface-cli login was done"
say "readiness ok"

# ---------- smoke gate ----------
if [[ "${DO_SMOKE}" == "1" ]]; then
  say "===== SMOKE (gate) ====="
  if bash smoke_test.sh; then
    say "smoke passed — proceeding to training"
  else
    say "❌ smoke FAILED — aborting before wasting GPU hours. Fix the bug and re-run."; exit 1
  fi
fi

# ---------- per-fold: train -> validate -> visuals ----------
for W in ${FOLDS}; do
  say "########## FOLD ${W} ##########"
  FOLD_DIR="${LOWO_ROOT}/fold_${W}"
  OUT_DIR="${RUNS_DIR}/fold_${W}"
  FOLD_CFG="${RUNS_DIR}/config_${W}.yaml"
  WEIGHTS="${OUT_DIR}/best_lora_weights.pt"

  # per-fold config: point data_dir + output_dir at this fold
  python3 - "${FULL_CONFIG}" "${FOLD_CFG}" "${FOLD_DIR}" "${OUT_DIR}" <<'PY'
import sys, yaml
full, out, dd, od = sys.argv[1:5]
c = yaml.safe_load(open(full))
c.setdefault("training", {})["data_dir"] = dd
c.setdefault("output", {})["output_dir"] = od
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(f"  config_{od}: data_dir={dd}")
PY

  if [[ "${DO_TRAIN}" == "1" ]]; then
    if [[ -f "${WEIGHTS}" ]]; then
      say "weights already exist for ${W} (skip train): ${WEIGHTS}"
    else
      say "training ${W} ..."
      python3 train_sam3_lora_native_claude.py --config "${FOLD_CFG}" --device ${DEVICE} \
        2>&1 | tee "${RESULTS_DIR}/${W}_train.log"
      rc=${PIPESTATUS[0]}
      if [[ ${rc} -ne 0 || ! -f "${WEIGHTS}" ]]; then
        say "❌ training ${W} failed (rc=${rc}, weights present=$([[ -f ${WEIGHTS} ]] && echo yes || echo no)) — skipping rest of this fold"
        backup_now; continue
      fi
      say "✅ trained ${W} -> ${WEIGHTS}"
    fi
  fi

  if [[ "${DO_VALIDATE}" == "1" && -f "${WEIGHTS}" ]]; then
    say "validating ${W} on held-out test ..."
    python3 validate_sam3_lora.py --config "${FOLD_CFG}" --weights "${WEIGHTS}" \
      --val_data_dir "${FOLD_DIR}/test" --prob_threshold "${THRESHOLD}" \
      2>&1 | tee "${RESULTS_DIR}/${W}_metrics.txt"
  fi

  if [[ "${DO_VISUALS}" == "1" && -f "${WEIGHTS}" ]]; then
    say "inference subset + compare for ${W} ..."
    PRED="${RUNS_DIR}/preds_${W}"
    IS_TEST_DIR="${FOLD_DIR}/test" IS_META_CSV="${META_CSV}" IS_PRED_DIR="${PRED}" \
      IS_REPO_DIR="${REPO_DIR}" IS_CONFIG="${FOLD_CFG}" IS_WEIGHTS="${WEIGHTS}" \
      IS_N_IMAGES="${N_IMAGES}" IS_WANT_VIEWS="${WANT_VIEWS}" IS_THRESHOLD="${THRESHOLD}" \
      python3 infer_subset.py 2>&1 | tee -a "${RESULTS_DIR}/${W}_infer.log" || say "⚠️ infer_subset issue (continuing)"
    CT_TEST_DIR="${FOLD_DIR}/test" CT_PRED_DIR="${PRED}" CT_OUT_DIR="${RESULTS_DIR}/compare_${W}" \
      python3 compare_test.py 2>&1 | tee -a "${RESULTS_DIR}/${W}_infer.log" || say "⚠️ compare issue (continuing)"
  fi

  # collect + back up after each fold
  [[ -f "${WEIGHTS}" ]] && cp -f "${WEIGHTS}" "${RESULTS_DIR}/${W}_best_lora_weights.pt" 2>/dev/null || true
  backup_now
  say "########## FOLD ${W} done ##########"
done

# ---------- wrap up ----------
say "===== ALL FOLDS DONE ====="
say "results in ${RESULTS_DIR}:"
ls -1 "${RESULTS_DIR}" | sed 's/^/    /'
backup_now

if [[ "${AUTO_STOP}" == "1" ]]; then
  if [[ ${backup_ok} -eq 1 && -n "${INSTANCE_ID}" ]] && command -v vastai >/dev/null 2>&1; then
    say "AUTO_STOP: results backed up — destroying instance ${INSTANCE_ID} in 60s (Ctrl-C to cancel)"
    sleep 60
    vastai destroy instance "${INSTANCE_ID}"
  else
    say "AUTO_STOP requested but skipped (need backup_ok=1, INSTANCE_ID set, vastai CLI). NOT destroying."
  fi
fi
say "pipeline finished."
