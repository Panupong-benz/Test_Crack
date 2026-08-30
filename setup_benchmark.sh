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
# ============ PASTE YOUR GOOGLE DRIVE LINK HERE ============
#
#   The file to upload is       03_annotation\_upload\pool_BM.zip
#                               (1,063,568,047 bytes, md5 d364d0e4...)
#   NOT                         03_annotation\_pool\POOL_BM.zip
#
# Those two names are one character apart and the wrong one was uploaded once
# already (Amendment A1.10): it is a hand-zipped copy of the pool FOLDER, so
# it has no pool/ + tools/ layout and cannot be used even if the md5 gate is
# bypassed. Check the size before uploading.
#
# Any of these forms works - the script tells them apart:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#   https://drive.google.com/drive/folders/<FOLDER_ID>
#   <FILE_ID>            (bare id, as before)
# Sharing must be "Anyone with the link", otherwise gdown saves an HTML
# permission page instead of the zip (the size gate below catches that).
# Leave empty to upload by hand instead: scp / vastai copy to $WORK.
#
POOL_GDRIVE="${POOL_GDRIVE:-}"
# ===========================================================
# backward-compatible alias: older notes/commands export POOL_GDRIVE_ID
POOL_GDRIVE="${POOL_GDRIVE:-${POOL_GDRIVE_ID:-}}"
POOLBM_MD5="${POOLBM_MD5:-d364d0e4f01406b7aadaed385e767663}"   # frozen A1.3
TEST_WALLS="${TEST_WALLS:-RW20,RW20C,RW20L,RW20T}"
TRAIN_ONLY="${TRAIN_ONLY:-RW40,N40,N20B}"
# ==================================================

die() { echo "FATAL: $*" >&2; exit 1; }
step() { echo -e "\n=== $* ==="; }

# ---- interpreter first (A1.13/A1.15): a fresh vast.ai tmux session does not
# activate the image venv, so the python3 on PATH may not be the one that has
# (or will get) torch. Find the torch-bearing python3 BEFORE anything runs,
# so every later step - and every pip install - hits the same interpreter.
find_torch_python() {
  local cand
  for cand in "$(command -v python3 2>/dev/null)" /venv/*/bin/python3 \
              /opt/conda/bin/python3 /usr/bin/python3; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import torch" >/dev/null 2>&1; then echo "$cand"; return 0; fi
  done
  return 1
}
TORCH_PY="$(find_torch_python || true)"
if [ -n "$TORCH_PY" ]; then
  export PATH="$(dirname "$TORCH_PY"):$PATH"
  echo "interpreter: $TORCH_PY (torch importable) - prepended to PATH"
else
  echo "interpreter: no python3 with torch found yet - setup_v2 will install it"
  echo "  into: $(command -v python3)"
fi

step "A. POOL -> folds (runs BEFORE setup_v2 so it finds them and skips gdown)"
if [ -d "$WORK/folds" ] || ls -d "$WORK"/fold_* >/dev/null 2>&1; then
  echo "folds already present - skipping pool build"
else
  # fetch the pool if it is not already on disk
  if [ ! -f "$POOL_ZIP" ] && [ -n "$POOL_GDRIVE" ]; then
    pip install -q -U gdown
    # A pasted link can be a FILE link, a FOLDER link, or a bare id. Tell
    # them apart instead of demanding one shape - pasting the folder link is
    # the natural thing to do and it used to fail silently (A1.10).
    src_kind="file"; src_id="$POOL_GDRIVE"
    case "$POOL_GDRIVE" in
      *"/folders/"*)
        src_kind="folder"
        src_id="${POOL_GDRIVE##*/folders/}"; src_id="${src_id%%[/?]*}" ;;
      *"/file/d/"*)
        src_id="${POOL_GDRIVE##*/file/d/}"; src_id="${src_id%%[/?]*}" ;;
      *"uc?id="*)
        src_id="${POOL_GDRIVE##*uc?id=}"; src_id="${src_id%%[&]*}" ;;
    esac
    echo "pool source: $src_kind  id=$src_id"
    if [ "$src_kind" = "folder" ]; then
      rm -rf "$WORK/_pooldl"
      gdown --folder "https://drive.google.com/drive/folders/$src_id" \
            -O "$WORK/_pooldl" || die "gdown --folder failed (is the folder
  shared as 'Anyone with the link'?)"
      found=$(find "$WORK/_pooldl" -name "pool_BM.zip" | head -1)
      if [ -z "$found" ]; then
        n=$(find "$WORK/_pooldl" -name "*.zip" | wc -l)
        [ "$n" = "1" ] || { ls -lR "$WORK/_pooldl"; die "expected pool_BM.zip
  in that folder; found $n zip(s). Upload 03_annotation/_upload/pool_BM.zip."; }
        found=$(find "$WORK/_pooldl" -name "*.zip" | head -1)
      fi
      mv "$found" "$POOL_ZIP" || die "could not move $found -> $POOL_ZIP"
    else
      # NOT `gdown --id`: that flag was REMOVED in gdown 5.x and the pip
      # line above always installs the newest one. uc?id= works on every
      # version; --fuzzy on the share URL is the fallback.
      gdown "https://drive.google.com/uc?id=$src_id" -O "$POOL_ZIP" || \
        gdown --fuzzy "https://drive.google.com/file/d/$src_id/view" -O "$POOL_ZIP" || \
        die "gdown failed - upload $POOL_ZIP to $WORK via Jupyter/scp and re-run"
    fi
  fi
  # HARD STOP: without the pool this used to fall through to setup_v2's gdown
  # of an OLD folds.zip, and check_folds never ran - the whole 181-job queue
  # would train on the wrong dataset in silence. Never again.
  [ -f "$POOL_ZIP" ] || die "no $POOL_ZIP and no POOL_GDRIVE set.
  Upload pool_BM.zip to $WORK (scp/vastai copy/Jupyter) or export
  POOL_GDRIVE='<drive link>'. Refusing to continue: the legacy folds
  fallback would silently train on the wrong dataset."

  # A 1 GB file goes through Drive's virus-scan interstitial. If sharing
  # is not "anyone with the link", gdown saves that HTML page AS the zip
  # and the md5 gate below then fails pointing at the wrong problem.
  bytes=$(stat -c%s "$POOL_ZIP" 2>/dev/null || echo 0)
  if [ "$bytes" -lt 100000000 ]; then
    echo "--- first 200 bytes of $POOL_ZIP ---"; head -c 200 "$POOL_ZIP"; echo
    die "$POOL_ZIP is only $bytes bytes (expected ~1064 MB).
  Almost always a Drive permission page, not a zip: set the file sharing
  to 'Anyone with the link' and re-run, or upload it by hand to $WORK."
  fi
  got=$(md5sum "$POOL_ZIP" | cut -d' ' -f1)
  echo "$(basename "$POOL_ZIP") md5 = $got   (expected: $POOLBM_MD5)"
  if [ "$POOLBM_MD5" != "TBD-at-freeze" ] && [ "$got" != "$POOLBM_MD5" ]; then
    if [ "$got" = "00a444f459ff36001cd46dc4daf12aef" ]; then
      die "md5 mismatch: this is _pool/POOL_BM.zip (the hand-zipped pool
  FOLDER), not the packed artifact. Upload 03_annotation/_upload/pool_BM.zip
  instead - 1,063,568,047 bytes, md5 $POOLBM_MD5 (Amendment A1.10)."
    fi
    die "md5 mismatch - wrong pool archive (Amendment A1 records the frozen
  one: 03_annotation/_upload/pool_BM.zip, 1,063,568,047 bytes)"
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
fi

# ---- data gates: OUTSIDE the build branch, so a re-run still gates ----
# The first real rental died at the image gate AFTER folds were written, so
# a re-run took the "folds already present" path and would have skipped both
# gates entirely - the failure would have vanished instead of being fixed
# (A1.11). ~2 s. The fold gate needs the frozen expectation from the pool
# archive; the image gate needs only the folds.
if [ -f "$WORK/pool_extract/folds_summary_expected.json" ]; then
  python3 "$REPO_DIR/benchmark/check_folds.py" \
      --expected "$WORK/pool_extract/folds_summary_expected.json" \
      --got "$WORK/folds/folds_summary.json" \
      || die "fold gate FAILED - the pool did not reproduce the frozen split"
else
  echo "fold gate: SKIPPED - folds already on disk; re-extract the pool to gate them"
fi
# image frame vs COCO record: PIL ignores EXIF orientation while Roboflow
# applied it, so a flagged photo would train/predict on the wrong axis
# (Amendment A1.4). ~1 s for 381 header reads.
# Pillow is the gate's ONLY third-party import and deps do not install
# until steps B/C, so a fresh instance reached this line with no PIL and
# the gate died on ModuleNotFoundError - reported as "image gate FAILED",
# i.e. a data verdict for an environment fault (A1.11). Ensure it here;
# the gate must stay in step A, before the expensive bootstrap.
python3 -c "import PIL" 2>/dev/null || pip install -q pillow \
    || die "could not install Pillow for the image gate"
python3 "$REPO_DIR/benchmark/check_images.py" --data "$WORK/folds" \
    || die "image gate FAILED - image frame does not match the COCO record"

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
# Pin the interpreter that actually has torch. A vast.ai image often keeps
# torch in a venv (/venv/main) that a FRESH tmux session does not activate,
# and some images have no `python` alias at all - so a queue launched from
# a reconnected session ran the SYSTEM python3 and every job died on
# "No module named 'torch'" while the smoke, run in the original session,
# had just used 17 GB of VRAM (A1.13). Resolve it once, here, where it is
# provably right, and let run_benchmark.sh put it on PATH for every job.
PYBIN="$(python3 -c "import sys, torch; sys.stdout.write(sys.executable)" 2>/dev/null)"
[ -n "$PYBIN" ] || die "the python3 on PATH cannot import torch - setup should have fixed this"
PYDIR="$(dirname "$PYBIN")"
cat > "$WORK/.bm_env" <<ENV
export nnUNet_raw="$nnUNet_raw"
export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
# interpreter pinned at setup time (A1.13); PATH first so both `python3`
# and any `python` shim in the same dir resolve to the torch-bearing one
export BM_PYBIN="$PYBIN"
export PATH="$PYDIR:$PATH"
ENV
echo "nnU-Net dirs + interpreter ($PYBIN) saved to $WORK/.bm_env"

step "F. selftests (final gate)"
python3 benchmark/eval_masks.py --selftest || die "eval_masks selftest FAILED"
# to_nnunet runs only in the LAST block of the queue (row A1), so a bad
# dataset.json or label range would surface after every training hour was
# already paid for (Amendment A1.5). Two seconds here instead.
python3 benchmark/to_nnunet.py --selftest || die "to_nnunet selftest FAILED"
# the image gate above runs on all-clean data, so a broken gate would pass
# silently. Its selftest plants a genuinely EXIF-flagged image and asserts
# raw 40x20 FAILS where oriented 20x40 passes - a green line here proves the
# gate can detect, not merely that it ran (A1.11).
python3 benchmark/check_images.py --selftest || die "check_images selftest FAILED"

echo -e "\nsetup_benchmark complete. Next - INTERIM SCOPE (A6 seed 0 + A5, Amendment A1.8):"
echo "  1. bash run_benchmark.sh smoke-a6     # ~15 min: pipe + s/step for the hour estimate"
echo "  2. source /workspace/.bm_env   # PATH + interpreter (needed in any NEW shell/tab)"
echo "  3. python3 benchmark/make_jobs.py --rows a6 a5 --seeds 0 --batch 8 --out jobs.yaml"
echo "  4. bash run_benchmark.sh full  # inside tmux; job progress streams to this screen"
echo ""
echo "  (the full six-row grid is SHELVED pending the advisor - Q18.4. To run it:"
echo "   bash run_benchmark.sh smoke, then make_jobs.py with no --rows/--seeds.)"
