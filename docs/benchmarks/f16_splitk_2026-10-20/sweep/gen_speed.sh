#!/bin/bash
# gen_speed.sh OUT_DIR K "tile;S;mode tile;S;mode ...": 512 x 512 x K kernels.
# tile is Y_CTA_OVERRIDE's six numbers or "auto" (the profile's choice); S = 0
# is the unsplit 3-parameter kernel; mode is f32, mma or default. Writes
# <tile>_s<S>_<mode>.ptx and prints the column specs for sbench.py.
set -e
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1; K=$2
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
mkdir -p $D; cp $S/xbar_ptx/final/.ysu_hw_profile $D/; cd $D
specs=""
for v in $3; do
  IFS=';' read t s mode <<< "$v"
  if [ "$s" = 0 ]; then nm=gemm_f16_512x512x$K; P=""; else nm=gemm_f16_512x512x${K}_splitk$s; P=", P: GlobalMemory<F32>"; fi
  printf '@tile(512, 512, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $K $nm "$P" > $nm.ysu
  out=${t//,/_}_s${s}_$mode.ptx
  envs=()
  [ "$t" != auto ] && envs+=(Y_CTA_OVERRIDE=$t)
  [ "$mode" != default ] && envs+=(Y_F16_ACCUMULATE=$mode)
  env "${envs[@]}" $X/target/release/Y $nm.ysu --emit-ptx -o $out > ${out%.ptx}.log 2>&1
  sp=$(ptxas -arch=sm_89 -v $out -o /dev/null 2>&1 | grep -A2 "entry function '$nm'" | grep -oE "[0-9]+ bytes spill stores" | grep -oE "^[0-9]+")
  [ "${sp:-0}" != 0 ] && echo "# $out spills $sp bytes" >&2
  specs="$specs ${out%.ptx}=$D/$out"
done
echo $specs
