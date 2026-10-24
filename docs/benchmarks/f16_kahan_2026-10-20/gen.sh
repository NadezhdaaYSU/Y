#!/bin/bash
# gen.sh OUT_DIR: <OUT_DIR>/<shape>/<column>.ptx for accuracy_kahan.py and speed.py,
# with the compiler in $X and the re-tuned profile the committed PTX came from.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
one() { # shape column splits mode tile
  local m n k; IFS=x read m n k <<< "$1"
  mkdir -p $D/$1; cd $D/$1; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  local nm=gemm_f16_${1}; P=""
  [ $3 != 0 ] && nm=${nm}_splitk$3 && P=", P: GlobalMemory<F32>"
  printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $m $n $k $nm "$P" > $2.ysu
  local envs=(Y_F16_ACCUMULATE=$4); [ $5 != auto ] && envs+=(Y_CTA_OVERRIDE=$5)
  env "${envs[@]}" $X/target/release/Y $2.ysu --emit-ptx -o $2.ptx > $2.log 2>&1 || echo "$1 $2: compile failed"
  echo "$1 $2: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $2.ptx | head -1) | $(ptxas -arch=sm_89 -v $2.ptx -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | head -2 | tr '\n' ' ')"
}
for sh in 1x4096x4096 8x4096x4096 32x4096x4096; do
  for mode in f32 kahan mma; do one $sh Y_$mode 0 $mode auto; done
done
one 512x512x4096 Y_f32 0 f32 auto; one 512x512x4096 Y_kahan 0 kahan auto
one 512x512x4096 Y_f32_s2 2 f32 64,64,32,4,2,4; one 512x512x4096 Y_kahan_s2 2 kahan 64,64,32,4,2,4
for s in 4 16; do
  one 512x512x65536 Y_f32_s$s $s f32 128,128,32,4,2,3; one 512x512x65536 Y_kahan_s$s $s kahan 128,128,32,4,2,3
done
