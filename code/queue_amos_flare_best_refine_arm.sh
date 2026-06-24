#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/ARM/logs/amos_flare_best_refine_arm.log
FLARE_BEST=/root/autodl-tmp/ARM/runs/strict_swinunetr_flare22_independent_arm_1k_100e_no_post_180e/checkpoint_best.pt
AMOS_BEST=/root/autodl-tmp/ARM/runs/strict_swinunetr_amos22_independent_arm_1k_100e_no_post_180e/checkpoint_best.pt

mkdir -p "$(dirname "$LOG")"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

last_epoch() {
  local metrics="$1"
  if [ ! -f "$metrics" ]; then
    echo 0
    return
  fi
  awk -F, 'NR>1 && $1 ~ /^[0-9]+$/ {e=$1} END {print e+0}' "$metrics"
}

run_flare_refine() {
  local out=/root/autodl-tmp/ARM/runs/refine_swinunetr_flare22_arm_best_80e
  mkdir -p "$out"
  echo "[start] FLARE22 ARM best-refine out=$out current_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
  "$PY" code/arm_runner/finetune_flare22_swinunetr.py \
    --data-root /root/autodl-tmp/ARM/data/flare22 \
    --output-dir "$out" \
    --epochs 80 \
    --eval-every 2 \
    --batch-size 1 \
    --num-workers 8 \
    --cache-rate 1.0 \
    --val-count 8 \
    --roi-size 96 96 96 \
    --pixdim 1.5 1.5 2.0 \
    --a-min -175 \
    --a-max 250 \
    --pos-ratio 4.0 \
    --neg-ratio 1.0 \
    --num-samples 6 \
    --sw-batch-size 1 \
    --overlap 0.625 \
    --loss dice_focal \
    --lambda-dice 1.0 \
    --lambda-focal 0.75 \
    --focal-gamma 2.0 \
    --lr 0.00002 \
    --weight-decay 0.000005 \
    --swin-feature-size 24 \
    --swin-depths 2 2 2 2 \
    --swin-num-heads 3 6 12 24 \
    --swin-window-size 7 7 7 \
    --swin-use-checkpoint \
    --num-classes 14 \
    --amp \
    --no-keep-largest \
    --sharing-strategy file_system \
    --init-checkpoint "$FLARE_BEST"
  echo "[done] FLARE22 ARM best-refine final_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
}

run_amos_refine() {
  local out=/root/autodl-tmp/ARM/runs/refine_swinunetr_amos22_arm_best_80e
  mkdir -p "$out"
  echo "[start] AMOS22 ARM best-refine out=$out current_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
  "$PY" code/arm_runner/finetune_amos22_swinunetr.py \
    --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
    --output-dir "$out" \
    --epochs 80 \
    --eval-every 2 \
    --batch-size 1 \
    --num-workers 8 \
    --cache-rate 0.75 \
    --val-count 8 \
    --roi-size 96 96 96 \
    --pixdim 1.5 1.5 2.0 \
    --a-min -175 \
    --a-max 250 \
    --pos-ratio 4.0 \
    --neg-ratio 1.0 \
    --num-samples 6 \
    --sw-batch-size 1 \
    --overlap 0.625 \
    --loss dice_focal \
    --lambda-dice 1.0 \
    --lambda-focal 0.75 \
    --focal-gamma 2.0 \
    --lr 0.00002 \
    --weight-decay 0.000005 \
    --swin-feature-size 24 \
    --swin-depths 2 2 2 2 \
    --swin-num-heads 3 6 12 24 \
    --swin-window-size 7 7 7 \
    --swin-use-checkpoint \
    --amp \
    --no-keep-largest \
    --sharing-strategy file_system \
    --init-checkpoint "$AMOS_BEST"
  echo "[done] AMOS22 ARM best-refine final_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
}

{
  echo "[queue] AMOS/FLARE ARM best-refine start $(timestamp)"
  echo "[queue] policy: refine from completed strict ARM best checkpoints; keep old strict runs untouched."
  run_flare_refine
  run_amos_refine
  echo "[queue] AMOS/FLARE ARM best-refine done $(timestamp)"
} >> "$LOG" 2>&1
