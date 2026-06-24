#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ARM

LOG=/root/autodl-tmp/ARM/logs/watch_lung_then_restart_arm_first_with_flare22.log
mkdir -p /root/autodl-tmp/ARM/logs /root/autodl-tmp/ARM/runs

timestamp() { date '+%F %T %Z'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

last_epoch() {
  local metrics="$1"
  if [ ! -f "$metrics" ]; then
    echo 0
    return 0
  fi
  awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ {e=$1} END {if (e == "") print 0; else print e}' "$metrics"
}

LUNG_RUN=/root/autodl-tmp/ARM/runs/strict_swinunetr_msd_lung_independent_arm_1k_100e_no_post_160e
NEW_QUEUE=code/arm_runner/queue_arm_first_no_amos_then_random_180e.sh

log "watcher start; waiting for Lung ARM to reach epoch 180 before restarting updated queue with FLARE22/MM-WHS"
while [ "$(last_epoch "$LUNG_RUN/metrics.csv")" -lt 180 ]; do
  log "lung_arm_epoch=$(last_epoch "$LUNG_RUN/metrics.csv")/180"
  sleep 60
done

log "Lung ARM complete enough; stopping old in-memory queue if alive"
pgrep -f "bash .*queue_arm_first_no_amos_then_random_180e.sh" | xargs -r kill -TERM || true
sleep 10
pgrep -f "bash .*queue_arm_first_no_amos_then_random_180e.sh" | xargs -r kill -KILL || true

log "waiting for any active finetune to exit"
while pgrep -f "code/arm_runner/(finetune_msd_binary_swinunetr.py|finetune_flare22_swinunetr.py|finetune_mmwhs_swinunetr.py)" >/dev/null 2>&1; do
  sleep 15
done

chmod +x "$NEW_QUEUE"
log "starting updated ARM-first queue with FLARE22/MM-WHS before Random"
nohup bash "$NEW_QUEUE" >/root/autodl-tmp/ARM/logs/queue_arm_first_no_amos_then_random_180e.nohup 2>&1 &
new_pid=$!
echo "$new_pid" > /root/autodl-tmp/ARM/runs/queue_arm_first_no_amos_then_random_180e.pid
log "new queue pid=${new_pid}"
