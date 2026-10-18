#!/bin/bash
# Wait (up to WAIT_S, default 1800 s) until the GPU's mean utilisation over 10
# samples is <= 5% with no game running (games.sh), then run "$@" holding
# gpu.lock. Prints the reading it started at; exits 3 if it never got quiet.
P=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
deadline=$(( $(date +%s) + ${WAIT_S:-1800} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  exec 9>"$P/gpu.lock"; flock 9
  games=$("$P/games.sh")
  mean=$(for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk '{s+=$1} END {printf "%.1f", s/NR}')
  if [ -z "$games" ] && awk "BEGIN{exit !($mean <= 5)}"; then
    echo "QUIET $(date +%T) mean ${mean}%"
    "$@"; exit $?
  fi
  flock -u 9; exec 9>&-; sleep 15
done
echo "NEVER QUIET"; exit 3
