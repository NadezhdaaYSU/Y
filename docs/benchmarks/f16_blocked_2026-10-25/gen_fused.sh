#!/bin/bash
# gen_fused.sh OUT: the committed fused SwiGLU (4096) and bias+ReLU (8192) kernels and the 16384^3 GEMM,
# as the chain (mma), blocked 128 and blocked 256, into OUT/<column>/ (bench_swiglu.py, bench_bias.py and
# bench_ab.py names), plus the chain GEMM at 4096 (g4096.ptx) for bench_swiglu.py's unfused column.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
for col in mma blk128 blk256; do
  mkdir -p $D/$col; cd $D/$col; cp $S/xbar_ptx/final/.ysu_hw_profile .
  case $col in mma) envs=(Y_F16_ACCUMULATE=mma);; blk*) envs=(Y_F16_ACCUMULATE=blocked Y_F16_BLOCK_K=${col#blk});; esac
  for k in gemm_f16_swiglu_4096 gemm_f16_bias_relu_8192 gemm_f16_16384 gemm_f16_4096; do
    cp $X/tests/$k.ysu .
    e=("${envs[@]}"); [ $k = gemm_f16_4096 ] && e=(Y_F16_ACCUMULATE=mma)
    env "${e[@]}" $X/target/release/Y $k.ysu --emit-ptx -o $k.ptx > $k.log 2>&1 || echo "$col $k: compile failed"
    echo "$col $k: $(grep -m1 -oE 'CTA [0-9x]+ (\(fixed\) )?\| [0-9x]+ warps' $k.ptx) | blocks $(grep -c F16_BLOCK_KEEP $k.ptx) | $(ptxas -arch=sm_89 -v $k.ptx -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | head -2 | tr '\n' ' ')"
  done
  ln -sf gemm_f16_16384.ptx g16384.ptx; ln -sf gemm_f16_4096.ptx g4096.ptx
done
