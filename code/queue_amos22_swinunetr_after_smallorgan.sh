#!/usr/bin/env bash
set -euo pipefail

arm_root=/root/autodl-tmp/ARM
log_dir="$arm_root/logs"
mkdir -p "$log_dir"

current_pid_file="$log_dir/run_monai_standard_amos22_smallorgan_resume.pid"
smallorgan_pattern="finetune_amos22_monai_standard.py.*probe_monai_standard_amos22_smallorgan_resume_80e"
next_log="$log_dir/run_monai_swinunetr_amos22_full_v2.log"
next_pid="$log_dir/run_monai_swinunetr_amos22_full_v2.pid"

if [[ -f "$next_pid" ]] && kill -0 "$(cat "$next_pid")" 2>/dev/null; then
  echo "SwinUNETR AMOS22 run is already active: $(cat "$next_pid")"
  exit 0
fi

while pgrep -f "$smallorgan_pattern" >/dev/null; do
  echo "$(date -Is) waiting for AMOS22 small-organ run matching: $smallorgan_pattern"
  sleep 300
done

cd "$arm_root"
nohup code/arm_runner/run_monai_swinunetr_amos22_full.sh > "$next_log" 2>&1 &
echo $! > "$next_pid"
echo "started SwinUNETR AMOS22 full run pid=$(cat "$next_pid")"
