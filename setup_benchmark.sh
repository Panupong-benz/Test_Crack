#!/usr/bin/env bash
# setup_benchmark.sh — vast.ai RTX 5090 bootstrap for benchmark rows A1–A6
# (docs: THESIS_crack_tool/docs/benchmark_protocol.md + vastai_runbook.md).
# Run from inside the cloned repo:
#   cd /workspace/Test_Crack && bash setup_benchmark.sh
#
# Builds ON TOP of setup_v2.sh (which it calls first): v2 does the GPU/cu128
# gate, torch-protected deps, HF login, folds download+verify. This script
# adds what the benchmark needs beyond a plain SAM3-LoRA run:
#   * benchmark deps (segmentation-models-pytorch, nnunetv2) — torch protected
#   * POOL_BM md5 gate (fill POOLBM_MD5 from Amendment A1 after the freeze)
#   * data/ symlink -> detected folds root (jobs.yaml uses data/fold_<wall>)
#   * nnU-Net env dirs (row A1)
#   * eval_masks selftest as the final gate
set -uo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace

# ===================== CONFIG =====================
# POOL_BM folds.zip: set FOLDS_GDRIVE_ID before running (new 7-wall pool,
# NOT the old 2-fold zip). md5 recorded in benchmark_protocol.md Amendment A1.
POOLBM_MD5="${POOLBM_MD5:-TBD-at-freeze}"
# ==================================================

die() { echo "FATAL: $*" >&2; exit 1; }
step() { echo -e "\n=== $* ==="; }

step "A. base bootstrap (setup_v2.sh: GPU gate / deps / HF / folds)"
bash "$REPO_DIR/setup_v2.sh" || die "setup_v2.sh failed"

step "B. POOL_BM md5 gate"
if [ -f "$WORK/folds.zip" ]; then
  got=$(md5sum "$WORK/folds.zip" | cut -d' ' -f1)
  echo "folds.zip md5 = $got   (expected: $POOLBM_MD5)"
  if [ "$POOLBM_MD5" != "TBD-at-freeze" ] && [ "$got" != "$POOLBM_MD5" ]; then
    die "md5 mismatch — wrong folds zip (Amendment A1 records the frozen one)"
  fi
  [ "$POOLBM_MD5" = "TBD-at-freeze" ] && \
    echo "WARN: POOLBM_MD5 not set — running unfrozen data (smoke only!)"
else
  echo "no $WORK/folds.zip (folds already extracted) — md5 gate skipped"
fi

step "C. benchmark deps (torch protected — never let pip touch the cu128 wheel)"
grep -viE '^(torch|torchvision|torchaudio)\b' benchmark/requirements-5090.txt \
  > /tmp/req_bm.txt || true
pip install -q -r /tmp/req_bm.txt
python3 - <<'PY' || exit 1
import torch
import segmentation_models_pytorch as smp
import nnunetv2  # noqa
import skimage, cv2, yaml, pandas  # noqa
assert torch.cuda.is_available(), "pip run REPLACED torch — reinstall cu128 wheel"
print("smp", smp.__version__, "| torch still", torch.__version__,
      "cuda", torch.version.cuda)
PY

step "D. data/ symlink for jobs.yaml"
LOWO_ROOT="$(cat "$WORK/.lowo_root" 2>/dev/null || true)"
[ -n "$LOWO_ROOT" ] && ls -d "$LOWO_ROOT"/fold_* >/dev/null 2>&1 || \
  die "no folds root recorded — setup_v2 step 3 should have written $WORK/.lowo_root"
ln -sfn "$LOWO_ROOT" "$REPO_DIR/data"
ls -d "$REPO_DIR"/data/fold_* || die "data/ symlink broken"

step "E. nnU-Net env (row A1)"
export nnUNet_raw="$WORK/nnUNet_raw"
export nnUNet_preprocessed="$WORK/nnUNet_preprocessed"
export nnUNet_results="$WORK/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
# persist for later shells / queue jobs
cat > "$WORK/.bm_env" <<ENV
export nnUNet_raw="$nnUNet_raw"
export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
ENV
echo "nnU-Net dirs exported + saved to $WORK/.bm_env"

step "F. evaluator selftest (final gate)"
python3 benchmark/eval_masks.py --selftest || die "eval_masks selftest FAILED"

echo -e "\nsetup_benchmark complete. Next (runbook SS4):"
echo "  1. bash run_benchmark.sh smoke        # Phase 4a (~1 h): env+50-step+batch sweep"
echo "  2. fill measured batch into: python benchmark/make_jobs.py --batch <B> --out jobs.yaml"
echo "  3. bash run_benchmark.sh full         # kill-gate first, then the whole grid"
