#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_swinunetr.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/swinunetr_amos22_full_dicefocal_180e_v4 \
  --epochs 180 \
  --eval-every 5 \
  --val-count 1 \
  --roi-size 96 96 64 \
  --pixdim 1.5 1.5 3.0 \
  --a-min -175 \
  --a-max 250 \
  --batch-size 1 \
  --num-workers 2 \
  --cache-rate 0.25 \
  --num-samples 4 \
  --pos-ratio 5 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.625 \
  --loss dice_focal \
  --lambda-dice 1.0 \
  --lambda-focal 0.75 \
  --focal-gamma 2.0 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --swin-feature-size 24 \
  --swin-depths 2 2 2 2 \
  --swin-num-heads 3 6 12 24 \
  --swin-window-size 7 7 7 \
  --swin-use-checkpoint \
  --amp \
  --no-keep-largest \
  --sharing-strategy file_system
