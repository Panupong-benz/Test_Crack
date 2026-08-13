#!/usr/bin/env bash
# run_all_v2.sh — the 2026-08 journal run: train v2 + FULL evaluation per fold.
#
# v2 = rank-32 LoRA, 24 epochs (recall-oriented; see full_lora_config_v2.yaml).
# The July rank-16 weights, if uploaded to /workspace/prev_weights/, are scored
# on the same test split as a "v1" benchmark row.
#
# Per fold (RW20 then RW20T):
#   train 24 epochs (skip if best_lora_weights.pt exists)
#   -> validate_sam3_lora (mAP + cgF1, LoRA)          [instance metrics]
#   -> infer_fused whole  on valid  -> threshold sweep -> t*
#   -> infer_fused whole  on test   -> eval at t*      [PixelIoU/Dice/clDice,
#                                                       per-drift, panels]
#   -> infer_fused tilemax+tilemean on test at t*      [SS7.1 max vs mean]
#   -> zero-shot SAM3 whole on test -> eval at t*      [benchmark row]
#   -> training curves PNG + bundle DOWNLOAD_fold_<X>.zip
#
# No automatic off-box backup (user downloads the zip via Jupyter after each
# fold). Rent ON-DEMAND, not interruptible.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace
LOWO_ROOT="$(cat $WORK/.lowo_root 2>/dev/null || echo $WORK/folds)"
FULL_CONFIG="$REPO_DIR/configs/full_lora_config_v2.yaml"   # v2: rank 32, see config comments
V1_CONFIG="$REPO_DIR/configs/full_lora_config.yaml"        # rank 16 - REQUIRED to load July weights
RUNS_DIR="$WORK/outputs/lowo"
RESULTS_DIR="$WORK/results"
FOLDS="${FOLDS:-RW20 RW20T}"
EPOCHS="${EPOCHS:-24}"
DEVICE="${DEVICE:-0}"
TILE_SIZE=1008
TILE_OVERLAP=0.30
META_CSV="$REPO_DIR/coco_with_meta.csv"

mkdir -p "$RUNS_DIR" "$RESULTS_DIR"
cd "$REPO_DIR"

banner() { echo -e "\n############################################################\n# $*\n############################################################"; }
die() { echo "FATAL: $*" >&2; exit 1; }

[ -d "$LOWO_ROOT" ] || die "LOWO_ROOT $LOWO_ROOT missing - run setup_v2.sh"

# VRAM-adaptive batch settings
MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
BS=2; GA=8
if [ "$MB" -lt 20000 ]; then BS=1; GA=16; fi

for FOLD in $FOLDS; do
  DATA_DIR="$LOWO_ROOT/fold_$FOLD"
  OUT_DIR="$RUNS_DIR/fold_$FOLD"
  RES="$RESULTS_DIR/fold_$FOLD"
  CFG="$RUNS_DIR/config_$FOLD.yaml"
  mkdir -p "$OUT_DIR" "$RES"
  [ -d "$DATA_DIR" ] || die "missing $DATA_DIR"

  banner "FOLD $FOLD - config"
  python3 - "$FULL_CONFIG" "$CFG" "$DATA_DIR" "$OUT_DIR" "$EPOCHS" "$BS" "$GA" <<'PY'
import sys
import yaml
src, dst, data_dir, out_dir, epochs, bs, ga = sys.argv[1:8]
cfg = yaml.safe_load(open(src))
cfg["training"]["data_dir"] = data_dir
cfg["training"]["num_epochs"] = int(epochs)
cfg["training"]["batch_size"] = int(bs)
cfg["training"]["gradient_accumulation_steps"] = int(ga)
cfg["training"]["checkpoint_path"] = None   # let HF resolve sam3.pt itself
cfg["output"]["output_dir"] = out_dir
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"wrote {dst}: data={data_dir} out={out_dir} epochs={epochs} bs={bs} ga={ga}")
PY

  if [ -f "$OUT_DIR/best_lora_weights.pt" ]; then
    banner "FOLD $FOLD - weights exist, skip training"
  else
    banner "FOLD $FOLD - TRAIN ($EPOCHS epochs)"
    python3 train_sam3_lora_native_claude.py --config "$CFG" --device "$DEVICE" \
      || die "training failed for $FOLD"
  fi
  W="$OUT_DIR/best_lora_weights.pt"
  [ -f "$W" ] || die "no best weights for $FOLD"

  banner "FOLD $FOLD - instance metrics (mAP/cgF1) on test"
  python3 validate_sam3_lora.py --config "$CFG" --weights "$W" \
    --val_data_dir "$DATA_DIR/test" --prob_threshold 0.3 \
    2>&1 | tee "$RES/validate_lora_test.log" || echo "WARN validate failed"

  banner "FOLD $FOLD - whole-image instances (valid + test)"
  python3 infer_fused.py --config "$CFG" --weights "$W" \
    --data_dir "$DATA_DIR/valid" --out_dir "$RES/preds_valid" \
    --variants whole --det-threshold 0.05 || die "infer valid failed"
  python3 infer_fused.py --config "$CFG" --weights "$W" \
    --data_dir "$DATA_DIR/test" --out_dir "$RES/preds_test" \
    --variants whole --det-threshold 0.05 || die "infer test failed"

  banner "FOLD $FOLD - threshold sweep on VALID"
  python3 eval_metrics.py --data_dir "$DATA_DIR/valid" \
    --npz "$RES/preds_valid/whole" --meta_csv "$META_CSV" \
    --out_dir "$RES/eval" --tag valid_sweep || die "sweep failed"
  TSTAR=$(python3 -c "import json;print(json.load(open('$RES/eval/chosen_threshold.json'))['threshold'])")
  echo "chosen threshold t* = $TSTAR"

  banner "FOLD $FOLD - TEST metrics at t* (LoRA whole)"
  python3 eval_metrics.py --data_dir "$DATA_DIR/test" \
    --npz "$RES/preds_test/whole" --threshold "$TSTAR" \
    --meta_csv "$META_CSV" --out_dir "$RES/eval" --tag test_lora \
    --val_stats "$OUT_DIR/val_stats.json" || echo "WARN test eval failed"

  banner "FOLD $FOLD - SS7.1 fusion (tilemax + tilemean) on test at t*"
  python3 infer_fused.py --config "$CFG" --weights "$W" \
    --data_dir "$DATA_DIR/test" --out_dir "$RES/preds_test" \
    --variants tilemax tilemean --tile-size $TILE_SIZE \
    --tile-overlap $TILE_OVERLAP --det-threshold "$TSTAR" \
    || echo "WARN fusion inference failed"
  for V in tilemax tilemean; do
    python3 eval_metrics.py --data_dir "$DATA_DIR/test" \
      --masks "$RES/preds_test/$V" --meta_csv "$META_CSV" \
      --out_dir "$RES/eval" --tag "test_$V" --panels 6 \
      || echo "WARN eval $V failed"
  done

  banner "FOLD $FOLD - ZERO-SHOT benchmark on test at t*"
  python3 infer_fused.py --config "$CFG" --base \
    --data_dir "$DATA_DIR/test" --out_dir "$RES/preds_test_base" \
    --variants whole --det-threshold 0.05 || echo "WARN base infer failed"
  python3 eval_metrics.py --data_dir "$DATA_DIR/test" \
    --npz "$RES/preds_test_base/whole" --threshold "$TSTAR" \
    --meta_csv "$META_CSV" --out_dir "$RES/eval" --tag test_zeroshot \
    --panels 6 || echo "WARN base eval failed"

  # v1 comparison: July rank-16 weights, uploaded manually via Jupyter to
  # /workspace/prev_weights/fold_<X>_v1.pt. Loaded with the OLD config (rank 16)
  # or the LoRA shapes will not match. Skipped silently when absent (RW20T
  # was never trained in July, so it has no v1).
  V1W="$WORK/prev_weights/fold_${FOLD}_v1.pt"
  if [ -f "$V1W" ]; then
    banner "FOLD $FOLD - v1 (July) weights on test at t*"
    python3 infer_fused.py --config "$V1_CONFIG" --weights "$V1W" \
      --data_dir "$DATA_DIR/test" --out_dir "$RES/preds_test_v1" \
      --variants whole --det-threshold 0.05 || echo "WARN v1 infer failed"
    python3 eval_metrics.py --data_dir "$DATA_DIR/test" \
      --npz "$RES/preds_test_v1/whole" --threshold "$TSTAR" \
      --meta_csv "$META_CSV" --out_dir "$RES/eval" --tag test_v1 \
      --panels 6 || echo "WARN v1 eval failed"
  fi

  banner "FOLD $FOLD - benchmark table"
  python3 - "$RES/eval" "$FOLD" <<'PY'
import json
import sys
from pathlib import Path
d = Path(sys.argv[1]); fold = sys.argv[2]
rows = []
for tag, label in [("test_zeroshot", "SAM3 zero-shot"),
                   ("test_v1", "LoRA v1 (Jul r16-18ep)"),
                   ("test_lora", "LoRA v2 (r32-24ep)"),
                   ("test_tilemax", "v2 + overlap max"),
                   ("test_tilemean", "v2 + overlap mean")]:
    p = d / f"summary_{tag}.json"
    if not p.exists():
        continue
    s = json.load(open(p))["mean"]
    rows.append((label, s))
print(f"\n=== {fold}: benchmark (test split, per-image mean) ===")
print(f"{'variant':<22}{'IoU':>7}{'Dice':>7}{'clDice':>8}{'Prec':>7}{'Rec':>7}")
for label, s in rows:
    print(f"{label:<22}{s['iou']:>7.3f}{s['dice']:>7.3f}"
          f"{s['cldice']:>8.3f}{s['precision']:>7.3f}{s['recall']:>7.3f}")
json.dump({l: s for l, s in rows}, open(d / "benchmark_table.json", "w"), indent=2)
PY

  banner "FOLD $FOLD - bundle"
  cp "$CFG" "$RES/" 2>/dev/null || true
  cp "$OUT_DIR/val_stats.json" "$RES/" 2>/dev/null || true
  cp "$OUT_DIR/best_lora_weights.pt" "$RES/" 2>/dev/null || true
  cp "$OUT_DIR/last_lora_weights.pt" "$RES/" 2>/dev/null || true
  ( cd "$RESULTS_DIR" && zip -q -r "$WORK/DOWNLOAD_fold_${FOLD}.zip" "fold_$FOLD" )
  banner "FOLD $FOLD DONE -> DOWNLOAD NOW: /workspace/DOWNLOAD_fold_${FOLD}.zip ($(du -h $WORK/DOWNLOAD_fold_${FOLD}.zip | cut -f1))"
done

banner "ALL FOLDS DONE - download every /workspace/DOWNLOAD_fold_*.zip then destroy the instance"
