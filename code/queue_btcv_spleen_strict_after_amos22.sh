#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

log=/root/autodl-tmp/ARM/logs/queue_btcv_spleen_strict_after_amos22.log
mkdir -p /root/autodl-tmp/ARM/logs

run_btcv() {
  local name="$1"
  local checkpoint="$2"
  local out="/root/autodl-tmp/ARM/runs/strict_swinunetr_btcv_${name}_no_post_160e"
  if [ -f "$out/checkpoint_last.pt" ]; then
    echo "[btcv] $name checkpoint exists"
    return 0
  fi
  echo "[btcv] $name start $(date)"
  cmd=(
    /root/miniconda3/bin/python code/arm_runner/finetune_btcv_swinunetr.py
    --data-root /root/autodl-tmp/ARM/data/btcv
    --output-dir "$out"
    --epochs 160
    --eval-every 5
    --batch-size 1
    --num-workers 2
    --cache-rate 1.0
    --val-count 4
    --roi-size 96 96 96
    --pixdim 1.5 1.5 2.0
    --a-min -175
    --a-max 250
    --pos-ratio 1.0
    --neg-ratio 1.0
    --num-samples 4
    --sw-batch-size 1
    --overlap 0.5
    --loss dice_focal
    --lambda-dice 1.0
    --lambda-focal 0.75
    --focal-gamma 2.0
    --lr 1e-4
    --weight-decay 1e-5
    --swin-feature-size 24
    --swin-depths 2 2 2 2
    --swin-num-heads 3 6 12 24
    --swin-window-size 7 7 7
    --swin-use-checkpoint
    --amp
    --no-keep-largest
    --sharing-strategy file_system
  )
  if [ -n "$checkpoint" ]; then
    cmd+=(--init-checkpoint "$checkpoint" --init-encoder-only --init-encoder-prefix swinViT)
  fi
  "${cmd[@]}"
  echo "[btcv] $name done $(date)"
}

run_spleen() {
  local name="$1"
  local checkpoint="$2"
  local out="/root/autodl-tmp/ARM/runs/strict_swinunetr_msd_spleen_${name}_no_post_300e"
  if [ -f "$out/checkpoint_last.pt" ]; then
    echo "[spleen] $name checkpoint exists"
    return 0
  fi
  echo "[spleen] $name start $(date)"
  cmd=(
    /root/miniconda3/bin/python code/arm_runner/finetune_msd_binary_swinunetr.py
    --data-root /root/autodl-tmp/ARM/data/msd/Task09_Spleen
    --output-dir "$out"
    --epochs 300
    --eval-every 5
    --batch-size 1
    --num-workers 2
    --cache-rate 1.0
    --val-count 8
    --roi-size 96 96 96
    --pixdim 1.5 1.5 2.0
    --a-min -175
    --a-max 250
    --pos-ratio 1.0
    --neg-ratio 1.0
    --num-samples 4
    --sw-batch-size 1
    --overlap 0.5
    --loss dice_focal
    --lambda-dice 1.0
    --lambda-focal 0.75
    --focal-gamma 2.0
    --lr 1e-4
    --weight-decay 1e-5
    --swin-feature-size 24
    --swin-depths 2 2 2 2
    --swin-num-heads 3 6 12 24
    --swin-window-size 7 7 7
    --swin-use-checkpoint
    --amp
    --no-keep-largest
    --sharing-strategy file_system
  )
  if [ -n "$checkpoint" ]; then
    cmd+=(--init-checkpoint "$checkpoint" --init-encoder-only --init-encoder-prefix swinViT)
  fi
  "${cmd[@]}"
  echo "[spleen] $name done $(date)"
}

run_family() {
  local dataset="$1"
  local fn="$2"
  "$fn" independent_arm_1k_100e /root/autodl-tmp/ARM/runs/independent_swinunetr_arm_1k_100e/checkpoint_last.pt
  "$fn" independent_random_1k_100e /root/autodl-tmp/ARM/runs/independent_swinunetr_random_1k_100e/checkpoint_last.pt
  "$fn" independent_topk_1k_100e /root/autodl-tmp/ARM/runs/independent_swinunetr_topk_1k_100e/checkpoint_last.pt
  "$fn" fromscratch ""
  echo "[$dataset] family done $(date)"
}

{
  echo "[watcher] start $(date)"
  while pgrep -f "queue_amos22_strict_after_mmwhs.sh" >/dev/null; do
    echo "[watcher] waiting for AMOS22 strict queue $(date)"
    sleep 300
  done
  while pgrep -f "finetune_amos22_swinunetr.py" >/dev/null; do
    echo "[watcher] waiting for AMOS22 finetune $(date)"
    sleep 300
  done

# BTCV is intentionally paused here. The current BTCV wrapper is binary
# (foreground/background) because it inherits the MSD binary trainer's label
# transform. It must not be used for the manuscript BTCV multi-organ Dice table.
run_family spleen run_spleen

  echo "[watcher] all done $(date)"
} >> "$log" 2>&1
