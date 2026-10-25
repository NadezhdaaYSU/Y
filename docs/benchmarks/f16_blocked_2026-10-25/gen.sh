#!/bin/bash
# gen.sh OUT_DIR: <OUT_DIR>/<shape>/<column>.ptx for the accuracy and speed harnesses,
# with the compiler in $X and the re-tuned profile the committed PTX came from.
# Columns: Y_mma (the chain), Y_default (no knob; with DEFAULT=1), Y_f32 (unless NO_F32=1),
# Y_blk<L> (blocked, L of K a block; BLOCKS lists the L).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
one() { # shape column envs...
  local sh=$1 col=$2; shift 2; local m n k; IFS=x read m n k <<< "$sh"
  mkdir -p $D/$sh; cd $D/$sh; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $sh > $col.ysu
  env "$@" $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1 || echo "$sh $col: compile failed"
  echo "$sh $col: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $col.ptx | head -1) | $(ptxas -arch=sm_89 -v $col.ptx -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | head -2 | tr '\n' ' ')"
}
for sh in ${SHAPES:-256x256x256 1024x1024x1024 4096x4096x4096 8192x8192x8192 512x512x1024 512x512x4096 512x512x16384 512x512x65536}; do
  one $sh Y_mma Y_F16_ACCUMULATE=mma
  [ -n "$DEFAULT" ] && one $sh Y_default Y_UNUSED=1
  [ -z "$NO_F32" ] && one $sh Y_f32 Y_F16_ACCUMULATE=f32
  for L in ${BLOCKS:-64 128 256 512 1024}; do one $sh Y_blk$L Y_F16_ACCUMULATE=blocked Y_F16_BLOCK_K=$L; done
done
