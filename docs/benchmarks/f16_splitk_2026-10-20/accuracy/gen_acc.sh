#!/bin/bash
# gen_acc.sh OUT_DIR: <OUT_DIR>/<K>/<column>.ptx for accuracy_split.py.
set -e
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
for K in 4096 16384 65536; do
  mkdir -p $D/$K; cd $D/$K; cp $S/xbar_ptx/final/.ysu_hw_profile .
  one() { # column split mode tile
    if [ $2 = 0 ]; then nm=gemm_f16_512x512x$K; P=""; else nm=gemm_f16_512x512x${K}_splitk$2; P=", P: GlobalMemory<F32>"; fi
    printf '@tile(512, 512, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $K $nm "$P" > $1.ysu
    envs=(Y_F16_ACCUMULATE=$3); [ $4 != auto ] && envs+=(Y_CTA_OVERRIDE=$4)
    env "${envs[@]}" $X/target/release/Y $1.ysu --emit-ptx -o $1.ptx > $1.log 2>&1
  }
  one Y_mma 0 mma auto; one Y_f32 0 f32 auto
  for s in 2 4 8 16; do one Y_f32_s$s $s f32 64,64,32,4,2,4; done
  one Y_f32_s4_t128 4 f32 128,128,32,4,2,3
  for s in 2 4 8; do one Y_mma_s$s $s mma 64,64,32,4,2,4; done
done
