#!/bin/bash
# For each sweep shape: the revised rule's `_splitk` kernel (auto2.ptx) and the column of the sweep it equals.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
P=$S/auto_ptx; Y=$S/xbar/target/release/Y; . $S/xbar/docs/benchmarks/f16_auto_splitk_2026-10-25/shapes.sh
for sh in $SHAPES_A $SHAPES_D $SHAPES_B; do
  cd $P/$sh; $Y auto.ysu --emit-ptx -o auto2.ptx > auto2.log 2>&1 || { echo "$sh COMPILE FAIL"; continue; }
  s=$(grep -m1 -oE 'in [0-9]+ splits?' auto2.ptx | grep -oE '[0-9]+')
  norm() { grep -v 'Y AUTO SPLIT-K' "$1" | sed -E "s/g_${sh}_splitk[0-9]*/NAME/g"; }
  twin=none
  if [ "$s" = 1 ]; then cmp -s auto2.ptx auto.ptx && twin=auto.ptx
  else for f in s${s}_*.ptx; do cmp -s <(norm auto2.ptx) <(norm $f) && twin=$f; done; fi
  echo "$sh S=$s twin=$twin | $(grep -m1 -oE 'S=[0-9]+ by the split-K rule: .*' auto2.ptx | cut -c1-170)"
done
