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
# We ship the POOL (one copy of every image), not the folds: each LOWO fold
# holds the whole pool, so four folds upload every photo four times (measured
# on POOL_D1: 614 MB pool vs 2456 MB of folds). lowo_split is deterministic,
# so the folds are rebuilt here with symlinks and verified by REPRODUCING
# folds_summary - a stronger gate than a zip checksum.
# md5 recorded in benchmark_protocol.md Amendment A1/A1.3.
POOL_ZIP="${POOL_ZIP:-$WORK/pool_BM.zip}"
# Google Drive file id of pool_BM.zip. Upload the zip to Drive ONCE; every
# re-rented instance then pulls it in ~1 min instead of re-uploading 1 GB.
POOL_GDRIVE_ID="${POOL_GDRIVE_ID:-}"
POOLBM_MD5="${POOLBM_MD5:-d364d0e4f01406b7aadaed385e767663}"   # frozen A1.3
TEST_WALLS="${TEST_WALLS:-RW20,RW20C,RW20L,RW20T}"
TRAIN_ONLY="${TRAIN_ONLY:-RW40,N40,N20B}"
# ==================================================

die() { echo "FATAL: $*" >&2; exit 1; }
step() { echo -e "\n=== $* ==="; }

step "A. POOL -> folds (runs BEFORE setup_v2 so it finds them and skips gdown)"
if [ -d "$WORK/folds" ] || ls -d "$WORK"/fold_* >/dev/null 2>&1; then
  echo "folds already present - skipping pool build"
else
  # fetch the pool if it is not already on disk
  if [ ! -f "$POOL_ZIP" ] && [ -n "$POOL_GDRIVE_ID" ]; then
    echo "downloading pool from Google Drive id $POOL_GDRIVE_ID"
    pip install -q -U gdown
    gdown --id "$POOL_GDRIVE_ID" -O "$POOL_ZIP" || \
      die "gdown failed - upload $POOL_ZIP to $WORK via Jupyter/scp and re-run"
  fi
  # HARD STOP: without the pool this used to fall through to setup_v2's gdown
  # of an OLD folds.zip, and check_folds never ran - the whole 177-job queue
  # would train on the wrong dataset in silence. Never again.
  [ -f "$POOL_ZIP" ] || die "no $POOL_ZIP and no POOL_GDRIVE_ID set.
  Upload pool_BM.zip to $WORK (scp/vastai copy/Jupyter) or export
  POOL_GDRIVE_ID=<drive file id>. Refusing to continue: the legacy folds
  fallback would silently train on the wrong dataset."

  got=$(md5sum "$POOL_ZIP" | cut -d' ' -f1)
  echo "$(basename "$POOL_ZIP") md5 = $got   (expected: $POOLBM_MD5)"
  if [ "$POOLBM_MD5" != "TBD-at-freeze" ] && [ "$got" != "$POOLBM_MD5" ]; then
    die "md5 mismatch - wrong pool archive (Amendment A1 records the frozen one)"
  fi
  [ "$POOLBM_MD5" = "TBD-at-freeze" ] && \
    echo "WARN: POOLBM_MD5 not set - running unfrozen data (smoke only!)"
  # python zipfile: works before apt has installed unzip
  python3 -m zipfile -e "$POOL_ZIP" "$WORK/pool_extract" || die "unzip failed"
  POOL="$WORK/pool_extract/pool"
  META=$(ls "$POOL"/coco_with_meta_*.csv 2>/dev/null | head -1)
  [ -n "$META" ] || die "no coco_with_meta_*.csv inside the pool archive"
  echo "rebuilding folds with symlinks (deterministic: SEED=42)"
  python3 "$WORK/pool_extract/tools/lowo_split.py" \
      --coco "$POOL/_annotations.coco.json" --img-dir "$POOL" \
      --meta "$META" --out "$WORK/folds" \
      --test-walls "$TEST_WALLS" --train-only "$TRAIN_ONLY" \
      --img-mode symlink || die "lowo_split failed"
  python3 "$REPO_DIR/benchmark/check_folds.py" \
      --expected "$WORK/pool_extract/folds_summary_expected.json" \
      --got "$WORK/folds/folds_summary.json" \
      || die "fold gate FAILED - the pool did not reproduce the frozen split"
  # image frame vs COCO record: PIL ignores EXIF orientation while Roboflow
  # applied it, so a flagged photo would train/predict on the wrong axis
  # (Amendment A1.4). ~1 s for 381 header reads.
  python3 "$REPO_DIR/benchmark/check_images.py" --data "$WORK/folds" \
      || die "image gate FAILED - image frame does not match the COCO record"
fi

step "B. base bootstrap (setup_v2.sh: GPU gate / deps / HF / dataset verify)"
bash "$REPO_DIR/setup_v2.sh" || die "setup_v2.sh failed"

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

step "F. selftests (final gate)"
python3 benchmark/eval_masks.py --selftest || die "eval_masks selftest FAILED"
# to_nnunet runs only in the LAST block of the queue (row A1), so a bad
# dataset.json or label range would surface after every training hour was
# already paid for (Amendment A1.5). Two seconds here instead.
python3 benchmark/to_nnunet.py --selftest || die "to_nnunet selftest FAILED"

echo -e "\nsetup_benchmark complete. Next (runbook SS4):"
echo "  1. bash run_benchmark.sh smoke        # Phase 4a (~1 h): env+50-step+batch sweep"
echo "  2. fill measured batch into: python benchmark/make_jobs.py --batch <B> --out jobs.yaml"
echo "  3. bash run_benchmark.sh full         # kill-gate first, then the whole grid"
