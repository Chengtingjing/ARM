#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

echo "[restart-corrected] start $(date)"

pkill -f "finetune_amos22_swinunetr.py" 2>/dev/null || true
pkill -f "queue_arm_only_after_topk_pretrain.sh" 2>/dev/null || true
sleep 5
pkill -9 -f "finetune_amos22_swinunetr.py" 2>/dev/null || true
pkill -9 -f "queue_arm_only_after_topk_pretrain.sh" 2>/dev/null || true

out="runs/strict_swinunetr_amos22_independent_arm_1k_100e_no_post_180e"
if [ -d "$out" ]; then
  backup="runs/strict_swinunetr_amos22_independent_arm_1k_100e_no_post_180e_pre_corrected_$(date +%Y%m%d_%H%M%S)"
  mv "$out" "$backup"
  echo "[restart-corrected] backed up active run to $backup"
fi

nohup bash code/arm_runner/queue_arm_only_after_topk_pretrain.sh \
  >/root/autodl-tmp/ARM/logs/queue_arm_only_after_topk_pretrain.corrected.out 2>&1 &
echo "[restart-corrected] watcher_pid=$!"
sleep 10

echo "[restart-corrected] processes:"
ps -eo pid,ppid,stat,etime,args \
  | grep -E "queue_arm_only_after_topk_pretrain.sh|finetune_amos22_swinunetr.py" \
  | grep -v grep || true

echo "[restart-corrected] latest log:"
tail -40 logs/queue_arm_only_after_topk_pretrain.log 2>/dev/null || true
