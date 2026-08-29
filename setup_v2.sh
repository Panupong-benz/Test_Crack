#!/usr/bin/env bash
# setup_v2.sh — vast.ai bootstrap for the 2026-08 journal run (RTX 5090 / cu128).
# Run from inside the cloned repo:   cd /workspace/Test_Crack && bash setup_v2.sh
#
# Replaces setup.sh. Differences that matter:
#   * torch/torchvision from the image are PROTECTED (requirements filtered)
#   * hard gate on torch.cuda + CUDA >= 12.8 for sm_120 + bitsandbytes import
#   * folds root auto-detected and exported to /workspace/.lowo_root
#   * dataset verified: files exist AND only the 'Crack' category is annotated
#   * no secrets written anywhere (HF_TOKEN stays an env var)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace
# DELIBERATELY EMPTY. The July 2-fold folds.zip id used to live here as a
# default, so a forgotten pool upload silently downloaded the WRONG dataset
# (setup_benchmark step A then never ran check_folds). Folds now come from
# pool_BM.zip via setup_benchmark.sh; set this only to resurrect the legacy
# path on purpose.
FOLDS_GDRIVE_ID="${FOLDS_GDRIVE_ID:-}"
FOLDS_ZIP="$WORK/folds.zip"

step() { echo -e "\n=== $* ==="; }
die() { echo "FATAL: $*" >&2; exit 1; }

step "0. GPU / torch / CUDA gate"
command -v nvidia-smi >/dev/null || die "no nvidia-smi"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 - <<'PY' || exit 1
import sys
import torch
ok = torch.cuda.is_available()
cu = torch.version.cuda or "none"
print(f"torch {torch.__version__}  cuda_available={ok}  cuda={cu}")
if not ok:
    sys.exit("torch.cuda.is_available() is False - wrong image")
major, minor = (int(x) for x in cu.split(".")[:2])
if (major, minor) < (12, 8):
    sys.exit(f"CUDA {cu} < 12.8 - RTX 5090 (sm_120) needs a cu128 image. "
             "Fix: pip install torch torchvision --index-url "
             "https://download.pytorch.org/whl/cu128")
cap = torch.cuda.get_device_capability(0)
print("compute capability:", cap)
torch.zeros(8, device="cuda").sum().item()  # actually launch a kernel
print("kernel launch OK")
PY

step "1. apt + python deps (torch protected)"
apt-get update -qq && apt-get install -y -qq unzip >/dev/null 2>&1 || true
grep -viE '^(torch|torchvision|torchaudio)\b' requirements.txt > /tmp/req_notorch.txt || true
pip install -q -r /tmp/req_notorch.txt
pip install -q bitsandbytes pycocotools tqdm pyyaml huggingface_hub gdown \
    safetensors einops matplotlib scikit-image opencv-python-headless
python3 - <<'PY' || exit 1
import torch, bitsandbytes as bnb, skimage, matplotlib, cv2  # noqa
print("bitsandbytes", bnb.__version__, "| torch still", torch.__version__,
      "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "pip run REPLACED torch - reinstall cu128 wheel"
PY

step "2. HF token"
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || \
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  echo "HF login attempted (token from env only)"
else
  echo "WARN: HF_TOKEN not set - facebook/sam3 download will fail. export HF_TOKEN=hf_..."
fi

step "3. folds.zip"
if [ ! -d "$WORK/folds" ] && ! ls -d "$WORK"/fold_* >/dev/null 2>&1; then
  if [ ! -f "$FOLDS_ZIP" ]; then
    [ -n "$FOLDS_GDRIVE_ID" ] || die "no folds and no FOLDS_GDRIVE_ID.
  For the benchmark run setup_benchmark.sh instead - it builds the folds from
  pool_BM.zip and gates them with check_folds.py. This legacy path is only for
  a deliberate re-download of an old folds.zip."
    pip install -q -U gdown
    gdown --id "$FOLDS_GDRIVE_ID" -O "$FOLDS_ZIP" || \
      die "gdown failed - upload folds.zip to $WORK via Jupyter and re-run"
  fi
  unzip -q -o "$FOLDS_ZIP" -d "$WORK/folds_extract"
fi
# auto-detect the folds root (zip layouts differ)
LOWO_ROOT=""
for cand in "$WORK/folds" "$WORK/folds_extract" "$WORK/folds_extract/folds" \
            "$WORK/folds_extract/Fold 4 walls" "$WORK"; do
  if ls -d "$cand"/fold_* >/dev/null 2>&1; then LOWO_ROOT="$cand"; break; fi
done
[ -n "$LOWO_ROOT" ] || die "no fold_* directory found after extraction"
echo "$LOWO_ROOT" > "$WORK/.lowo_root"
echo "LOWO_ROOT = $LOWO_ROOT  ->  saved to $WORK/.lowo_root"
ls -d "$LOWO_ROOT"/fold_*

step "4. dataset verification"
python3 - "$LOWO_ROOT" <<'PY' || exit 1
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
bad = 0
for fold in sorted(root.glob("fold_*")):
    for split in ("train", "valid", "test"):
        j = fold / split / "_annotations.coco.json"
        if not j.exists():
            print(f"MISSING {j}"); bad += 1; continue
        d = json.load(open(j))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        used = {cats.get(a["category_id"], "?") for a in d["annotations"]}
        missing = [im["file_name"] for im in d["images"]
                   if not (fold / split / im["file_name"]).exists()]
        n_noncrack = sum(1 for a in d["annotations"]
                         if cats.get(a["category_id"], "").lower() != "crack")
        print(f"{fold.name}/{split}: {len(d['images'])} imgs, "
              f"{len(d['annotations'])} anns, cats used={sorted(used)}, "
              f"missing files={len(missing)}, non-Crack anns={n_noncrack}")
        bad += len(missing)
        if n_noncrack:
            print(f"  WARN: {n_noncrack} non-Crack annotations - the trainer "
                  "creates a query per category; folds should be Crack-only")
if bad:
    sys.exit(f"{bad} problems - fix the dataset before training")
print("dataset OK")
PY

step "5. VRAM-based config hint"
MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU memory: ${MB} MiB"
if [ "$MB" -lt 20000 ]; then
  echo "WARN: <20GB VRAM - run_all_v2.sh will use batch_size=1 / grad_accum=16"
fi

mkdir -p "$WORK/prev_weights"

echo -e "\nsetup_v2 complete. Next:"
echo "  1. (optional, for the v1 benchmark row) upload July weights via Jupyter to:"
echo "       /workspace/prev_weights/fold_RW20_v1.pt"
echo "  2. bash smoke_test.sh          # gate: rank-32 VRAM + 1-epoch sanity"
echo "  3. nohup bash run_all_v2.sh > /workspace/run.log 2>&1 &"
