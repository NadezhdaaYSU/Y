#!/bin/bash
# gen_acc_bk.sh OUT: compensated block folds (Y_F16_ACCUMULATE=blocked_kahan) against the default blocked sum, at
# the rule's three splits and at 512 x 512 x 4096 unsplit: Y_mma (the control), Y_unsplit and Y_auto (the default
# sums), Y_unsplit_bk and Y_auto_bk (the same kernels with compensated folds); with SHORT=1 also Y_*_bk64 and
# Y_*_bk128, compensated blocks of 64 and 128 (Y_F16_BLOCK_K).
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; O=$1; mkdir -p $O
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
one() { # shape column name params envs...
  local sh=$1 col=$2 nm=$3 extra=$4 m n k; shift 4; IFS=x read m n k <<< "$sh"
  mkdir -p $O/$sh; cd $O/$sh; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $m $n $k $nm "$extra" > $col.ysu
  env "$@" $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1 || echo "$sh $col: COMPILE FAILED"
  echo "$sh $col: $(grep -m1 -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $col.ptx) | $(grep -m1 -oE 'in [0-9]+ splits?' $col.ptx) | sub.rn $(grep -c sub.rn.f32 $col.ptx) | $(ptxas -arch=sm_89 -v $col.ptx -o /dev/null 2>&1 | grep -A3 "function '$nm'" | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' ')"
}
P=', P: GlobalMemory<F32>'
for sh in 512x512x4096 512x512x16384 512x512x65536 256x256x16384; do
  one $sh Y_mma g_$sh "" Y_F16_ACCUMULATE=mma
  one $sh Y_unsplit g_$sh "" Y_UNUSED=1
  one $sh Y_auto g_${sh}_splitk "$P" Y_UNUSED=1
  one $sh Y_unsplit_bk g_$sh "" Y_F16_ACCUMULATE=blocked_kahan
  one $sh Y_auto_bk g_${sh}_splitk "$P" Y_F16_ACCUMULATE=blocked_kahan
  [ -n "$SHORT" ] && for L in 64 128; do
    one $sh Y_auto_bk$L g_${sh}_splitk "$P" Y_F16_ACCUMULATE=blocked_kahan Y_F16_BLOCK_K=$L
    one $sh Y_unsplit_bk$L g_$sh "" Y_F16_ACCUMULATE=blocked_kahan Y_F16_BLOCK_K=$L
  done
done
