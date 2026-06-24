#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_monai_standard.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_multiclass_60e \
  --max-train 48 \
  --max-val 12 \
  --epochs 60 \
  --eval-every 5 \
  --val-count 1 \
  --roi-size 80 80 64 \
  --pixdim 2.0 2.0 3.0 \
  --a-min -175 \
  --a-max 250 \
  --batch-size 1 \
  --num-workers 2 \
  --cache-rate 1.0 \
  --num-samples 3 \
  --pos-ratio 4 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.5 \
  --loss dice_ce \
  --channels 16 32 64 128 256 \
  --strides 2 2 2 2 \
  --lr 2e-4 \
  --amp \
  --sharing-strategy file_system
