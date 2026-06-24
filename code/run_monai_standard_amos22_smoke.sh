#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_monai_standard.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/smoke_monai_standard_amos22_multiclass \
  --max-train 8 \
  --max-val 4 \
  --epochs 1 \
  --eval-every 1 \
  --val-count 1 \
  --roi-size 64 64 64 \
  --pixdim 2.0 2.0 3.0 \
  --a-min -175 \
  --a-max 250 \
  --batch-size 1 \
  --num-workers 0 \
  --cache-rate 0.25 \
  --num-samples 2 \
  --pos-ratio 3 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.5 \
  --loss dice_ce \
  --channels 16 32 64 128 256 \
  --strides 2 2 2 2 \
  --lr 2e-4 \
  --amp \
  --no-keep-largest \
  --sharing-strategy file_system
