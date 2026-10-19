#!/bin/bash
# regen.sh <out-dir> [env assignments...]: compiles the 27 committed GEMM
# kernels' sources with the profile they were generated under and compares
# each with the committed tests/<name>.ptx.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar
O=$1; shift
mkdir -p "$O"; cp $S/xbar_ptx/final/.ysu_hw_profile "$O/"
same=0; diff=0
for k in $(cat $S/swz/kernels.txt); do
  cp "$X/tests/$k.ysu" "$O/"
  (cd "$O" && env "$@" "$X/target/release/Y" "$k.ysu" --emit-ptx -o "$k.ptx" > "$k.log" 2>&1) || { echo "COMPILE FAIL $k"; continue; }
  if cmp -s "$O/$k.ptx" "$X/tests/$k.ptx"; then same=$((same+1)); else diff=$((diff+1)); echo "differs: $k"; fi
done
echo "same $same differ $diff"
