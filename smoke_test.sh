#!/usr/bin/env bash
# smoke_test.sh — fast sanity run BEFORE committing to the full training.
#
# Builds a tiny fold (a handful of images) from an existing fold, runs ONE epoch,
# and reports: did it crash? peak VRAM vs total (does it fit?), measured it/s, and
# an extrapolated full-run time. Because every step processes the same 1008px tile
# at the same batch size, the per-step VRAM and it/s measured here match the real
# run — only the number of steps differs. So this finishes in ~1-2 min but the
# numbers are representative.
#
#   export HF_TOKEN="hf_..."          # trainer needs it to fetch SAM3 base weights
#   ./smoke_test.sh
# (or in a notebook cell:  !bash smoke_test.sh )

set -uo pipefail

# ===================== CONFIG (edit) =====================
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
# v2 config (rank 32) so the smoke gates VRAM at the value the real run will use
# MUST match make_jobs.py's --base-config default, or the smoke hour gates
# VRAM and extrapolates hours for a model the queue never runs (v2 is rank 32
# / 24 epochs; the benchmark runs rank 16 / 30 epochs). Amendment A1.4.
FULL_CONFIG="${FULL_CONFIG:-${REPO_DIR}/configs/full_lora_config.yaml}"
LOWO_ROOT="$(cat /workspace/.lowo_root 2>/dev/null || echo /workspace/folds)"
SRC_FOLD="${SRC_FOLD:-${LOWO_ROOT}/fold_RW20}" # an existing fold (has train/ valid/ test/)
SMOKE_DIR="/workspace/smoke"              # scratch dir for the smoke fold + config
N_TRAIN=8                                 # images sampled into the smoke train split
N_EVAL=3                                  # images for valid and for test (each)
TIMEOUT=900                               # hard cap (s) in case it hangs
# A1.27: GPU ids for the trainer. "0" = single GPU (unchanged). "0 1" = the
# trainer self-launches torchrun on both; grad accumulation is divided by
# the count so the effective batch stays what the real queue uses.
DEVICES="${BM_DEVICES:-0}"
N_GPU=$(echo ${DEVICES} | wc -w)
# A1.28 item 157(c): at 4 ranks N_TRAIN=8 leaves only ~13 micro-steps per
# rank, so the s/it that the WHOLE rental is extrapolated from would be
# dominated by warmup and cudnn.benchmark autotune. Widen the smoke fold
# when it would otherwise be too short to measure. 1x/2x keep 8 exactly,
# so the A1.27 gate does not move. The extrapolation normalises by
# N_TRAIN (tiles_per_img below), so a wider smoke fold does not bias the
# hours estimate - it only makes the rate less noisy.
if (( N_GPU >= 4 )); then N_TRAIN=$(( 4 * N_GPU )); fi
N_TRAIN="${BM_SMOKE_TRAIN:-$N_TRAIN}"
# ========================================================

say() { echo "[$(date +%H:%M:%S)] $*"; }
SMOKE_FOLD="${SMOKE_DIR}/smoke_fold"
SMOKE_CFG="${SMOKE_DIR}/smoke_config.yaml"
LOG="${SMOKE_DIR}/smoke.log"

# ---------- pre-flight ----------
cd "${REPO_DIR}" || { echo "❌ REPO_DIR not found: ${REPO_DIR}"; exit 1; }
[[ -f train_sam3_lora_native_claude.py ]] || { echo "❌ trainer not in ${REPO_DIR}"; exit 1; }
[[ -f "${FULL_CONFIG}" ]]                  || { echo "❌ config not found: ${FULL_CONFIG}"; exit 1; }
[[ -f "${SRC_FOLD}/train/_annotations.coco.json" ]] || { echo "❌ ${SRC_FOLD}/train/_annotations.coco.json missing"; exit 1; }
[[ -n "${HF_TOKEN:-}" ]] || say "⚠️ HF_TOKEN not set — trainer may fail to fetch SAM3 weights"
mkdir -p "${SMOKE_DIR}"
rm -rf "${SMOKE_FOLD}"

# ---------- build the tiny smoke fold ----------
say "building smoke fold (train=${N_TRAIN}, valid/test=${N_EVAL}) from ${SRC_FOLD}"
python3 - "$SRC_FOLD" "$SMOKE_FOLD" "$N_TRAIN" "$N_EVAL" <<'PY'
import sys, os, json, shutil
src, dst, ntr, nev = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def subset(split, n):
    sdir = os.path.join(src, split); ddir = os.path.join(dst, split)
    os.makedirs(ddir, exist_ok=True)
    coco = json.load(open(os.path.join(sdir, "_annotations.coco.json")))
    imgs = coco["images"][:n]
    keep = {im["id"] for im in imgs}
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    for im in imgs:
        s = os.path.join(sdir, im["file_name"])
        if os.path.exists(s): shutil.copy2(s, os.path.join(ddir, im["file_name"]))
    out = {"images": imgs, "annotations": anns, "categories": coco["categories"]}
    for k in ("info", "licenses"):
        if k in coco: out[k] = coco[k]
    json.dump(out, open(os.path.join(ddir, "_annotations.coco.json"), "w"))
    print(f"  {split}: {len(imgs)} imgs / {len(anns)} anns")
subset("train", ntr)
subset("valid", nev)
subset("test",  nev)
PY
[[ -f "${SMOKE_FOLD}/train/_annotations.coco.json" ]] || { echo "❌ smoke fold build failed"; exit 1; }

# ---------- write a 1-epoch smoke config ----------
say "writing smoke config (num_epochs=1, no checkpointing)"
python3 - "$FULL_CONFIG" "$SMOKE_CFG" "$SMOKE_FOLD" "$N_GPU" <<'PY'
import sys, yaml
full, out, fold, ngpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = yaml.safe_load(open(full))
t = cfg.setdefault("training", {})
t["data_dir"]   = fold
t["num_epochs"] = 1
if ngpu > 1:   # A1.27 item 150: same rule as make_jobs.override_keys
    acc = int(t.get("gradient_accumulation_steps", 1))
    assert acc % ngpu == 0, f"accum {acc} not divisible by {ngpu} GPUs"
    t["gradient_accumulation_steps"] = acc // ngpu
    print(f"  {ngpu} GPUs -> gradient_accumulation_steps {acc} -> {acc // ngpu}")
t["save_steps"] = 10**9          # effectively never mid-epoch checkpoint
t["save_total_limit"] = 1
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print("  data_dir ->", fold)
PY

# ---------- VRAM monitor (background) ----------
VRAM_FILE="${SMOKE_DIR}/vram_peak.txt"; echo 0 > "${VRAM_FILE}"
have_smi=0; command -v nvidia-smi >/dev/null 2>&1 && have_smi=1
mon_pid=""
if [[ $have_smi -eq 1 ]]; then
  ( peak=0
    while true; do
      # max over cards (A1.27): the busiest GPU is the OOM question
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
      [[ "$u" =~ ^[0-9]+$ ]] && (( u > peak )) && { peak=$u; echo "$peak" > "${VRAM_FILE}"; }
      sleep 1
    done ) & mon_pid=$!
  TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
else
  say "⚠️ nvidia-smi missing — skipping VRAM check"; TOTAL_VRAM=""
fi
stop_mon() { [[ -n "${mon_pid}" ]] && { pkill -P "${mon_pid}" 2>/dev/null; kill "${mon_pid}" 2>/dev/null; } || true; }
trap stop_mon EXIT INT TERM

# ---------- run one epoch ----------
say "running 1 epoch (timeout ${TIMEOUT}s) — log: ${LOG}"
echo "============================================================"
start=$(date +%s)
say "devices: ${DEVICES} (${N_GPU} GPU)"
timeout "${TIMEOUT}" python3 train_sam3_lora_native_claude.py --config "${SMOKE_CFG}" --device ${DEVICES} 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "============================================================"
stop_mon

# ---------- report ----------
echo ""; say "================= SMOKE RESULT ================="
if [[ ${rc} -eq 124 ]]; then
  say "❌ TIMEOUT after ${TIMEOUT}s — likely a hang (data loader / download). See ${LOG}"; exit 1
elif [[ ${rc} -ne 0 ]]; then
  say "❌ trainer exited ${rc} — there IS a bug. Last lines:"; tail -n 15 "${LOG}"; exit ${rc}
fi
say "✅ ran to completion, no crash"

# it/s + step count from tqdm (handles both 'it/s' and 's/it')
its=$(grep -oE '[0-9]+(\.[0-9]+)?it/s' "${LOG}" | tail -1 | grep -oE '[0-9.]+')
spit=$(grep -oE '[0-9]+(\.[0-9]+)?s/it' "${LOG}" | tail -1 | grep -oE '[0-9.]+')
steps=$(grep -oE '[0-9]+/[0-9]+ \[' "${LOG}" | tail -1 | grep -oE '[0-9]+/[0-9]+')
if [[ -z "${its}" && -n "${spit}" ]]; then
  its=$(python3 -c "print(round(1/${spit},3))" 2>/dev/null)
fi
[[ -n "${its}" ]] && say "measured speed : ${its} it/s" || say "speed: (couldn't parse it/s — check ${LOG} tail)"
[[ -n "${steps}" ]] && say "smoke steps    : ${steps} (this tiny fold)"

# VRAM verdict
if [[ $have_smi -eq 1 ]]; then
  peak=$(cat "${VRAM_FILE}")
  if [[ -n "${TOTAL_VRAM}" && "${peak}" =~ ^[0-9]+$ ]]; then
    pct=$(python3 -c "print(round(100*${peak}/${TOTAL_VRAM}))" 2>/dev/null)
    say "peak VRAM      : ${peak} / ${TOTAL_VRAM} MiB (${pct}%)"
    (( pct >= 92 )) && say "   ⚠️ very close to limit — full run may OOM; lower batch_size or tile overlap"
  fi
fi

# exact full-run projection (A1.30 item 167): the GPU measures s/it, the
# step counts are COUNTED by the production TiledCOCODataset over each
# fold's COCO json - never extrapolated from the smoke sample again. The
# old tiles-per-image scaling under-read fold_RW20 by 2x because POOL_BM
# mixes 6-tile resized frames with 35-tile full-res frames, and whichever
# 8 images the smoke drew decided the whole rental's estimate.
if [[ -n "${its:-}" ]]; then
  python3 - "$its" "$SRC_FOLD" "$FULL_CONFIG" "$N_GPU" <<'PY'
import sys
from pathlib import Path
its = float(sys.argv[1]); src = Path(sys.argv[2])
full = Path(sys.argv[3]); ngpu = int(sys.argv[4])
sys.path.insert(0, "benchmark")
import count_tiles as ct                                   # noqa: E402
tcfg = ct.tiling_cfg(full)
batch, epochs = ct.cfg_batch_epochs(full)
s_it = 1.0 / max(its, 1e-6)
root = src.parent if src.parent != Path(".") else Path(".")
total_h, rows = 0.0, []
for fold in sorted(root.glob("fold_*")):
    if not (fold / "train" / "_annotations.coco.json").exists():
        continue
    try:
        ds = ct.build(fold, tcfg)
    except Exception as e:                                  # noqa: BLE001
        print(f"[extrapolate] WARN {fold.name}: {e}")
        continue
    tiles = len(ds.tile_specs)
    spr = ct.steps_per_rank(tiles, batch, ngpu)
    h = spr * epochs * s_it / 3600.0
    total_h += h
    rows.append((fold.name, tiles, spr, h))
if not rows:
    print("[extrapolate] WARNING: no fold_*/ with a COCO found - cannot "
          "project the run")
for name, tiles, spr, h in rows:
    print(f"[extrapolate] {name}: {tiles} tiles -> {spr} steps/epoch/GPU "
          f"x {epochs} ep on {ngpu} GPU -> ~{h:.1f} h  (COUNTED, not "
          f"extrapolated - A1.30)")
if rows:
    print(f"[extrapolate] ALL {len(rows)} folds: ~{total_h:.1f} h of a6 "
          f"training at {s_it:.2f} s/it  (+ a5/predict/eval ~1-2 h; "
          f"train only the folds you queue)")
PY
fi
say "==============================================="
say "if this looks good (A1.8 interim):"
say "  source /workspace/.bm_env"
say "  python3 benchmark/make_jobs.py --rows a6 a5 --seeds 0 --batch 8 --out jobs.yaml"
say "  bash run_benchmark.sh full"
