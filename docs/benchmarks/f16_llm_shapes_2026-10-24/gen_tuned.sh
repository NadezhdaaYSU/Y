#!/bin/bash
# gen_tuned.sh OUT SIZES...: as gen.sh, but `--autotune`: each unseen shape is measured on this GPU
# and recorded in OUT's copy of the profile (never the working one).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; O=$1; shift
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
mkdir -p $O; cd $O; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
for size in "$@"; do
  IFS=x read m n k <<< "$size"
  printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $size > g$size.ysu
  $X/target/release/Y g$size.ysu --emit-ptx --autotune -o g$size.ptx > g$size.log 2>&1 || echo "$size: compile failed"
  echo "$size: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' g$size.ptx | head -1) s$(grep -oE 'stages[^0-9]*[0-9]+' g$size.ptx | head -1 | grep -oE '[0-9]+$')"
done
