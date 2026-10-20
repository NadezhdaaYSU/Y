#!/bin/bash
# regen_all.sh OUT BIN: every tests/*.ysu with a committed .ptx, compiled by BIN
# with the re-tuned profile; prints the files whose output differs from the commit.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; O=$1; B=$2
mkdir -p $O; cp $S/xbar_ptx/final/.ysu_hw_profile $O/
unset Y_CTA_OVERRIDE Y_F16_ACCUMULATE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE
same=0; diff=0; fail=0
for p in $X/tests/*.ptx; do k=$(basename $p .ptx); [ -f $X/tests/$k.ysu ] || continue
  cp $X/tests/$k.ysu $O/
  (cd $O && timeout 120 "$B" $k.ysu --emit-ptx -o $k.ptx > $k.log 2>&1) || { fail=$((fail+1)); echo "fail $k"; continue; }
  if cmp -s $O/$k.ptx $p; then same=$((same+1)); else diff=$((diff+1)); echo "differs $k"; fi
done
echo "same $same differ $diff fail $fail"
