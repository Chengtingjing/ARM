#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

log=/root/autodl-tmp/ARM/logs/queue_amos22_strict_after_mmwhs.log
mkdir -p /root/autodl-tmp/ARM/logs

run_amos22() {
  local name="$1"
  local checkpoint="$2"
  local out="/root/autodl-tmp/ARM/runs/strict_swinunetr_amos22_${name}_no_post_180e"

  if [ -f "$out/checkpoint_last.pt" ]; then
    echo "[amos22] $name checkpoint exists"
    return 0
  fi

  echo "[amos22] $name start $(date)"
  cmd=(
    /root/miniconda3/bin/python code/arm_runner/finetune_amos22_swinunetr.py
    --data-root /root/autodl-tmp/ARM/data/amos22/extracted
    --output-dir "$out"
    --epochs 180
    --eval-every 5
    --val-count 1
    --roi-size 96 96 64
    --pixdim 1.5 1.5 3.0
    --a-min -175
    --a-max 250
    --batch-size 1
    --num-workers 2
    --cache-rate 0.25
    --num-samples 4
    --pos-ratio 5
    --neg-ratio 1
    --sw-batch-size 1
    --overlap 0.625
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
  echo "[amos22] $name done $(date)"
}

{
  echo "[watcher] start $(date)"
  while pgrep -f "queue_independent_random_topk_1k_100e" >/dev/null; do
    echo "[watcher] waiting for MM-WHS random/topk queue $(date)"
    sleep 300
  done
  while pgrep -f "queue_fromscratch_after_random_topk.sh" >/dev/null; do
    echo "[watcher] waiting for MM-WHS from-scratch watcher $(date)"
    sleep 300
  done
  while pgrep -f "finetune_mmwhs_swinunetr.py" >/dev/null; do
    echo "[watcher] waiting for MM-WHS finetune $(date)"
    sleep 300
  done

  run_amos22 \
    independent_arm_1k_100e \
    /root/autodl-tmp/ARM/runs/independent_swinunetr_arm_1k_100e/checkpoint_last.pt
  run_amos22 \
    independent_random_1k_100e \
    /root/autodl-tmp/ARM/runs/independent_swinunetr_random_1k_100e/checkpoint_last.pt
  run_amos22 \
    independent_topk_1k_100e \
    /root/autodl-tmp/ARM/runs/independent_swinunetr_topk_1k_100e/checkpoint_last.pt
  run_amos22 fromscratch ""

  echo "[watcher] all done $(date)"
} >> "$log" 2>&1
