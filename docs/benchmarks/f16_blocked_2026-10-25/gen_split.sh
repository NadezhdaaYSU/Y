#!/bin/bash
# gen_split.sh OUT: 512 x 512 x K split-K kernels (`_splitk<S>`) on f16_kahan_2026-10-20's tiles, each split
# float32 per mma (the default), chained (mma), or blocked (128, 256): OUT/<shape>/<column>.ptx.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
one() { # shape splits tile column envs...
  local sh=$1 s=$2 tile=$3 col=$4; shift 4; local m n k; IFS=x read m n k <<< "$sh"
  mkdir -p $D/$sh; cd $D/$sh; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s_splitk%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, P: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $sh $s > $col.ysu
  env Y_CTA_OVERRIDE=$tile "$@" $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1 || echo "$sh $col: compile failed"
  echo "$sh $col: $(grep -m1 -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $col.ptx) | blocks $(grep -c F16_BLOCK_KEEP $col.ptx) | $(ptxas -arch=sm_89 -v $col.ptx -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | head -2 | tr '\n' ' ')"
}
for sh in 512x512x16384 512x512x65536; do
  t=128,128,32,4,2,3
  one $sh 4 $t Y_f32_s4 Y_UNUSED=1
  one $sh 4 $t Y_mma_s4 Y_F16_ACCUMULATE=mma
  for L in 128 256; do one $sh 4 $t Y_blk${L}_s4 Y_F16_ACCUMULATE=blocked Y_F16_BLOCK_K=$L; done
done
