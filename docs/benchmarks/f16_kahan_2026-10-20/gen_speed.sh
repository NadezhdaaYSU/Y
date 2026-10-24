#!/bin/bash
# gen_speed.sh OUT_DIR: <OUT_DIR>/<mode>/g<size>.ptx (kernel gemm_f16_<size>) for
# ../f16_square_2026-10-18/bench_ab.py, each mode under Y_F16_ACCUMULATE=<mode>, with the
# compiler in $X and the re-tuned profile the committed PTX came from.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
for mode in f32 kahan mma; do
  mkdir -p $D/$mode; cd $D/$mode; cp $S/xbar_ptx/final/.ysu_hw_profile .
  for size in 1x4096x4096 4x4096x4096 8x4096x4096 32x4096x4096; do
    IFS=x read m n k <<< "$size"
    printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $size > g$size.ysu
    Y_F16_ACCUMULATE=$mode $X/target/release/Y g$size.ysu --emit-ptx -o g$size.ptx > g$size.log 2>&1 || echo "$mode $size: compile failed"
    echo "$mode $size: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' g$size.ptx | head -1) | $(ptxas -arch=sm_89 -v g$size.ptx -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' ')"
  done
done
