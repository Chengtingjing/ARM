#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_monai_standard.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_dicefocal_resume_80e \
  --max-train 96 \
  --max-val 24 \
  --init-checkpoint /root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_multiclass_60e/checkpoint_best.pt \
  --epochs 80 \
  --eval-every 5 \
  --val-count 1 \
  --roi-size 96 96 64 \
  --pixdim 1.5 1.5 3.0 \
  --a-min -175 \
  --a-max 250 \
  --batch-size 1 \
  --num-workers 2 \
  --cache-rate 1.0 \
  --num-samples 4 \
  --pos-ratio 5 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.625 \
  --loss dice_focal \
  --lambda-dice 1.0 \
  --lambda-focal 0.75 \
  --focal-gamma 2.0 \
  --channels 16 32 64 128 256 \
  --strides 2 2 2 2 \
  --lr 1e-4 \
  --amp \
  --sharing-strategy file_system
