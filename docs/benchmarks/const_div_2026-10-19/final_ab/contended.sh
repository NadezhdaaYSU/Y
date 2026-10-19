#!/bin/bash
# contended.sh: the final A/B (runs 3 and 4) and the ncu source-counter run, holding gpu.lock
# but NOT waiting for a quiet GPU: the user's game and other processes hold it. Each run's
# 10-sample mean utilisation before it starts is logged as "CONTENDED".
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
A=$SP/cdiv/ab; G2=$SP/cdiv/ncu_final
exec 9>"$SP/gpu.lock"; flock 9
mean() { for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; sleep 0.1; done | awk '{s+=$1} END {printf "%.1f", s/NR}'; }
for t in 3 4; do
  echo "$t: CONTENDED $(date +%T) mean $(mean)% games: $("$SP/games.sh" | grep -oE 'isaac-ng.exe|sober|AppId=[0-9]+' | sort -u | tr '\n' ' ')" >> $A/queue.txt
  bash $A/run.sh $t
done
echo "ncu: CONTENDED $(date +%T) mean $(mean)% (--clock-control none)" >> $G2/queue.txt
bash $G2/ncu_src.sh
