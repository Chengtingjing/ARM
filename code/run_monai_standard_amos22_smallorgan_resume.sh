#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_monai_standard.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_smallorgan_resume_80e \
  --max-train 144 \
  --max-val 36 \
  --init-checkpoint /root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_dicefocal_resume_80e/checkpoint_best.pt \
  --epochs 80 \
  --eval-every 5 \
  --val-count 1 \
  --roi-size 112 112 80 \
  --pixdim 1.25 1.25 2.5 \
  --a-min -175 \
  --a-max 250 \
  --batch-size 1 \
  --num-workers 2 \
  --cache-rate 1.0 \
  --num-samples 5 \
  --pos-ratio 7 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.75 \
  --loss tversky_ce \
  --lambda-tversky 1.0 \
  --lambda-ce 0.5 \
  --tversky-alpha 0.3 \
  --tversky-beta 0.7 \
  --channels 16 32 64 128 256 \
  --strides 2 2 2 2 \
  --lr 8e-5 \
  --amp \
  --no-keep-largest \
  --sharing-strategy file_system
