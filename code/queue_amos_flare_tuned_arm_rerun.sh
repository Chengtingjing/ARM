#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/ARM/logs/amos_flare_tuned_arm_rerun.log
ARM_CKPT=/root/autodl-tmp/ARM/runs/independent_swinunetr_arm_1k_100e/checkpoint_last.pt

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

has_active_train() {
  ps -eo pid,cmd \
    | grep -E "finetune_.*swinunetr|pretrain_swinunetr_mim" \
    | grep -v grep \
    | grep -v "queue_amos_flare_tuned_arm_rerun" \
    >/dev/null
}

wait_until_no_active_train() {
  echo "[wait] waiting for active training to finish $(timestamp)"
  while has_active_train; do
    ps -eo pid,etime,cmd \
      | grep -E "finetune_.*swinunetr|pretrain_swinunetr_mim" \
      | grep -v grep \
      | head -6 || true
    sleep 120
  done
  echo "[wait] GPU training slot is free $(timestamp)"
}

run_flare22_tuned() {
  local out=/root/autodl-tmp/ARM/runs/tuned_swinunetr_flare22_independent_arm_1k_100e_240e
  mkdir -p "$out"
  echo "[start] FLARE22 ARM tuned out=$out current_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
  "$PY" code/arm_runner/finetune_flare22_swinunetr.py \
    --data-root /root/autodl-tmp/ARM/data/flare22 \
    --output-dir "$out" \
    --epochs 240 \
    --eval-every 5 \
    --batch-size 1 \
    --num-workers 8 \
    --cache-rate 1.0 \
    --val-count 8 \
    --roi-size 96 96 96 \
    --pixdim 1.5 1.5 2.0 \
    --a-min -175 \
    --a-max 250 \
    --pos-ratio 6.0 \
    --neg-ratio 1.0 \
    --num-samples 8 \
    --sw-batch-size 1 \
    --overlap 0.625 \
    --loss dice_ce \
    --lambda-dice 1.0 \
    --lambda-ce 1.0 \
    --lr 0.00008 \
    --weight-decay 0.00001 \
    --swin-feature-size 24 \
    --swin-depths 2 2 2 2 \
    --swin-num-heads 3 6 12 24 \
    --swin-window-size 7 7 7 \
    --swin-use-checkpoint \
    --num-classes 14 \
    --amp \
    --no-keep-largest \
    --sharing-strategy file_system \
    --init-checkpoint "$ARM_CKPT" \
    --init-encoder-only \
    --init-encoder-prefix swinViT \
    --auto-resume
  echo "[done] FLARE22 ARM tuned final_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
}

run_amos22_tuned() {
  local out=/root/autodl-tmp/ARM/runs/tuned_swinunetr_amos22_independent_arm_1k_100e_240e
  mkdir -p "$out"
  echo "[start] AMOS22 ARM tuned out=$out current_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
  "$PY" code/arm_runner/finetune_amos22_swinunetr.py \
    --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
    --output-dir "$out" \
    --epochs 240 \
    --eval-every 5 \
    --batch-size 1 \
    --num-workers 8 \
    --cache-rate 0.75 \
    --val-count 8 \
    --roi-size 96 96 96 \
    --pixdim 1.5 1.5 2.0 \
    --a-min -175 \
    --a-max 250 \
    --pos-ratio 6.0 \
    --neg-ratio 1.0 \
    --num-samples 6 \
    --sw-batch-size 1 \
    --overlap 0.625 \
    --loss dice_ce \
    --lambda-dice 1.0 \
    --lambda-ce 1.0 \
    --lr 0.00008 \
    --weight-decay 0.00001 \
    --swin-feature-size 24 \
    --swin-depths 2 2 2 2 \
    --swin-num-heads 3 6 12 24 \
    --swin-window-size 7 7 7 \
    --swin-use-checkpoint \
    --amp \
    --no-keep-largest \
    --sharing-strategy file_system \
    --init-checkpoint "$ARM_CKPT" \
    --init-encoder-only \
    --init-encoder-prefix swinViT \
    --auto-resume
  echo "[done] AMOS22 ARM tuned final_epoch=$(last_epoch "$out/metrics.csv") $(timestamp)"
}

{
  echo "[queue] tuned AMOS/FLARE ARM rerun start $(timestamp)"
  echo "[queue] policy: do not interrupt active training; run FLARE22 tuned first, then AMOS22 tuned."
  wait_until_no_active_train
  run_flare22_tuned
  run_amos22_tuned
  echo "[queue] tuned AMOS/FLARE ARM rerun done $(timestamp)"
} >> "$LOG" 2>&1
