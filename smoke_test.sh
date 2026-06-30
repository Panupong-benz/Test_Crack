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
REPO_DIR="/Test_Crack"
FULL_CONFIG="${REPO_DIR}/configs/full_lora_config.yaml"
SRC_FOLD="/workspace/folds/fold_RW20"     # an existing fold (has train/ valid/ test/)
SMOKE_DIR="/workspace/smoke"              # scratch dir for the smoke fold + config
N_TRAIN=8                                 # images sampled into the smoke train split
N_EVAL=3                                  # images for valid and for test (each)
TIMEOUT=900                               # hard cap (s) in case it hangs
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
python3 - "$FULL_CONFIG" "$SMOKE_CFG" "$SMOKE_FOLD" <<'PY'
import sys, yaml
full, out, fold = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(full))
t = cfg.setdefault("training", {})
t["data_dir"]   = fold
t["num_epochs"] = 1
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
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
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
timeout "${TIMEOUT}" python3 train_sam3_lora_native_claude.py --config "${SMOKE_CFG}" 2>&1 | tee "${LOG}"
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

# rough full-run extrapolation
if [[ -n "${its:-}" && -n "${steps:-}" ]]; then
  python3 - "$its" "$steps" "$N_TRAIN" "$SRC_FOLD" "$FULL_CONFIG" <<'PY'
import sys, json, os, yaml
its=float(sys.argv[1]); smoke_steps=int(sys.argv[2].split('/')[-1])
ntr=int(sys.argv[3]); src=sys.argv[4]; full=sys.argv[5]
cfg=yaml.safe_load(open(full)); t=cfg.get("training",{})
bs=t.get("batch_size",2); epochs=t.get("num_epochs",30)
real=len(json.load(open(os.path.join(src,"train","_annotations.coco.json")))["images"])
tiles_per_img = smoke_steps*bs/max(ntr,1)
real_steps = tiles_per_img*real/bs
sec = real_steps*epochs/max(its,1e-6)
print(f"[extrapolate] ~{tiles_per_img:.1f} tiles/img -> ~{real_steps:.0f} steps/epoch x {epochs} ep")
print(f"[extrapolate] this fold (~{real} imgs): ~{sec/60:.0f} min  (~{sec/3600:.1f} h)  at {its} it/s")
print(f"[extrapolate] both folds ~= roughly double (RW20T is larger).")
PY
fi
say "==============================================="
say "if this looks good: ./run_train.sh RW20"
