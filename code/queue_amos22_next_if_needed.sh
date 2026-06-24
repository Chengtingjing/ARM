#!/usr/bin/env bash
set -euo pipefail

arm_root="/root/autodl-tmp/ARM"
code_dir="$arm_root/code/arm_runner"
log_dir="$arm_root/logs"
mkdir -p "$log_dir"
cd "$arm_root"

status_log="$log_dir/queue_amos22_next_if_needed_$(date +%Y%m%d_%H%M%S).log"
echo "$(date -Is) AMOS22 queue check" | tee "$status_log"

active_train="$(pgrep -af 'finetune_amos22_swinunetr.py|run_monai_swinunetr_amos22_full|run_monai_swinunetr_amos22_v[56789]' || true)"
if [[ -n "$active_train" ]]; then
  echo "$(date -Is) active AMOS22 training detected; not launching another run" | tee -a "$status_log"
  echo "$active_train" | tee -a "$status_log"
  exit 0
fi

full_run="$arm_root/runs/swinunetr_amos22_full_dicefocal_180e_v4"
v7_run="$arm_root/runs/swinunetr_amos22_full_smallorgan_tversky_resume_100e_v7"
v8_run="$arm_root/runs/swinunetr_amos22_full_balanced_tversky_resume_120e_v8"
v9_run="$arm_root/runs/swinunetr_amos22_fastval_tversky_resume_80e_v9"
v6_run="$arm_root/runs/swinunetr_amos22_full_smallorgan_tversky_resume_100e_v6"
v5_run="$arm_root/runs/swinunetr_amos22_full_tversky_resume_120e_v5"
if [[ ! -f "$full_run/metrics.csv" || ! -f "$full_run/checkpoint_best.pt" ]]; then
  echo "$(date -Is) full SwinUNETR evidence missing; no resume launch" | tee -a "$status_log"
  exit 0
fi

best="$(/root/miniconda3/bin/python - "$full_run/metrics.csv" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
best = -1.0
best_epoch = ''
with path.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        try:
            val = float(row.get('val_dice', -1))
        except Exception:
            continue
        if val > best:
            best = val
            best_epoch = row.get('epoch', '')
print(f'{best_epoch},{best}')
PY
)"
best_epoch="${best%%,*}"
best_dice="${best#*,}"
echo "$(date -Is) full SwinUNETR best_epoch=$best_epoch best_dice=$best_dice" | tee -a "$status_log"

should_launch="$(/root/miniconda3/bin/python - "$best_dice" <<'PY'
import sys
val = float(sys.argv[1])
if val <= 1:
    val *= 100
print('yes' if val < 87.22 else 'no')
PY
)"

if [[ "$should_launch" != "yes" ]]; then
  echo "$(date -Is) AMOS22 full SwinUNETR is within tolerance; no Tversky resume needed" | tee -a "$status_log"
  exit 0
fi

if [[ -f "$v7_run/STOPPED_TOO_SLOW.txt" ]]; then
  echo "$(date -Is) small-organ Tversky v7 was stopped as too slow; considering v8 fallback" | tee -a "$status_log"
elif [[ -f "$v7_run/checkpoint_best.pt" || -f "$v7_run/metrics.csv" ]]; then
  echo "$(date -Is) small-organ Tversky v7 already exists; not relaunching" | tee -a "$status_log"
  exit 0
fi

if [[ ! -f "$v7_run/STOPPED_TOO_SLOW.txt" && -x "$code_dir/run_monai_swinunetr_amos22_v7_smallorgan_tversky_resume.sh" ]]; then
  echo "$(date -Is) launching AMOS22 SwinUNETR small-organ Tversky/CE v7 resume" | tee -a "$status_log"
  nohup bash "$code_dir/run_monai_swinunetr_amos22_v7_smallorgan_tversky_resume.sh" > "$log_dir/run_monai_swinunetr_amos22_v7_smallorgan_tversky_resume.log" 2>&1 &
  echo $! > "$log_dir/run_monai_swinunetr_amos22_v7_smallorgan_tversky_resume.pid"
  echo "$(date -Is) started pid=$(cat "$log_dir/run_monai_swinunetr_amos22_v7_smallorgan_tversky_resume.pid")" | tee -a "$status_log"
  exit 0
fi

if [[ -f "$v8_run/STOPPED_TOO_SLOW.txt" ]]; then
  echo "$(date -Is) balanced Tversky v8 was stopped as too slow; considering v9 fast-val fallback" | tee -a "$status_log"
elif [[ -f "$v8_run/checkpoint_best.pt" || -f "$v8_run/metrics.csv" ]]; then
  echo "$(date -Is) balanced Tversky v8 already exists; not relaunching" | tee -a "$status_log"
  exit 0
fi

if [[ ! -f "$v8_run/STOPPED_TOO_SLOW.txt" && -x "$code_dir/run_monai_swinunetr_amos22_v8_balanced_tversky_resume.sh" ]]; then
  echo "$(date -Is) launching AMOS22 SwinUNETR balanced Tversky/CE v8 resume" | tee -a "$status_log"
  nohup bash "$code_dir/run_monai_swinunetr_amos22_v8_balanced_tversky_resume.sh" > "$log_dir/run_monai_swinunetr_amos22_v8_balanced_tversky_resume.log" 2>&1 &
  echo $! > "$log_dir/run_monai_swinunetr_amos22_v8_balanced_tversky_resume.pid"
  echo "$(date -Is) started pid=$(cat "$log_dir/run_monai_swinunetr_amos22_v8_balanced_tversky_resume.pid")" | tee -a "$status_log"
  exit 0
fi

if [[ -f "$v9_run/checkpoint_best.pt" || -f "$v9_run/metrics.csv" ]]; then
  echo "$(date -Is) fast-val Tversky v9 already exists; not relaunching" | tee -a "$status_log"
  exit 0
fi

if [[ -x "$code_dir/run_monai_swinunetr_amos22_v9_fastval_tversky_resume.sh" ]]; then
  echo "$(date -Is) launching AMOS22 SwinUNETR fast-val Tversky/CE v9 resume" | tee -a "$status_log"
  nohup bash "$code_dir/run_monai_swinunetr_amos22_v9_fastval_tversky_resume.sh" > "$log_dir/run_monai_swinunetr_amos22_v9_fastval_tversky_resume.log" 2>&1 &
  echo $! > "$log_dir/run_monai_swinunetr_amos22_v9_fastval_tversky_resume.pid"
  echo "$(date -Is) started pid=$(cat "$log_dir/run_monai_swinunetr_amos22_v9_fastval_tversky_resume.pid")" | tee -a "$status_log"
  exit 0
fi

if [[ -f "$v6_run/checkpoint_best.pt" || -f "$v6_run/metrics.csv" ]]; then
  echo "$(date -Is) small-organ Tversky v6 already exists; not relaunching" | tee -a "$status_log"
  exit 0
fi

if [[ -x "$code_dir/run_monai_swinunetr_amos22_v6_smallorgan_tversky_resume.sh" ]]; then
  echo "$(date -Is) launching AMOS22 SwinUNETR small-organ Tversky/CE v6 resume" | tee -a "$status_log"
  nohup bash "$code_dir/run_monai_swinunetr_amos22_v6_smallorgan_tversky_resume.sh" > "$log_dir/run_monai_swinunetr_amos22_v6_smallorgan_tversky_resume.log" 2>&1 &
  echo $! > "$log_dir/run_monai_swinunetr_amos22_v6_smallorgan_tversky_resume.pid"
  echo "$(date -Is) started pid=$(cat "$log_dir/run_monai_swinunetr_amos22_v6_smallorgan_tversky_resume.pid")" | tee -a "$status_log"
  exit 0
fi

if [[ -f "$v5_run/checkpoint_best.pt" || -f "$v5_run/metrics.csv" ]]; then
  echo "$(date -Is) Tversky v5 already exists; not relaunching" | tee -a "$status_log"
  exit 0
fi

echo "$(date -Is) launching AMOS22 SwinUNETR Tversky/CE v5 resume" | tee -a "$status_log"
nohup bash "$code_dir/run_monai_swinunetr_amos22_v5_tversky_resume.sh" > "$log_dir/run_monai_swinunetr_amos22_v5_tversky_resume.log" 2>&1 &
echo $! > "$log_dir/run_monai_swinunetr_amos22_v5_tversky_resume.pid"
echo "$(date -Is) started pid=$(cat "$log_dir/run_monai_swinunetr_amos22_v5_tversky_resume.pid")" | tee -a "$status_log"
