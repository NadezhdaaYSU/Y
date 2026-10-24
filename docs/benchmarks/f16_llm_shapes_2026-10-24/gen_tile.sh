#!/bin/bash
# gen_tile.sh OUT TILE SIZES...: g<size>.ptx under Y_CTA_OVERRIDE=TILE (cta_m,cta_n,cta_k,warps_m,warps_n,stages).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; O=$1; T=$2; shift 2
unset Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
mkdir -p $O; cd $O; cp $S/xbar_ptx/final/.ysu_hw_profile .
for size in "$@"; do
  IFS=x read m n k <<< "$size"
  printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $size > g$size.ysu
  Y_CTA_OVERRIDE=$T $X/target/release/Y g$size.ysu --emit-ptx -o g$size.ptx > g$size.log 2>&1 || echo "$T $size: compile failed"
  grep -q "CTA ${T%%,*}x" g$size.ptx || echo "$T $size: override not taken: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' g$size.ptx | head -1)"
done
