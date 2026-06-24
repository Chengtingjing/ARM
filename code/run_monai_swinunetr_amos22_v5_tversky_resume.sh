#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM/code/arm_runner

/root/miniconda3/bin/python finetune_amos22_swinunetr.py \
  --data-root /root/autodl-tmp/ARM/data/amos22/extracted \
  --output-dir /root/autodl-tmp/ARM/runs/swinunetr_amos22_full_tversky_resume_120e_v5 \
  --init-checkpoint /root/autodl-tmp/ARM/runs/swinunetr_amos22_full_dicefocal_180e_v4/checkpoint_best.pt \
  --epochs 120 \
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
  --pos-ratio 6 \
  --neg-ratio 1 \
  --sw-batch-size 1 \
  --overlap 0.75 \
  --loss tversky_ce \
  --lambda-tversky 1.0 \
  --lambda-ce 0.3 \
  --tversky-alpha 0.25 \
  --tversky-beta 0.75 \
  --lr 5e-5 \
  --weight-decay 1e-5 \
  --swin-feature-size 24 \
  --swin-depths 2 2 2 2 \
  --swin-num-heads 3 6 12 24 \
  --swin-window-size 7 7 7 \
  --swin-use-checkpoint \
  --amp \
  --keep-largest \
  --sharing-strategy file_system
