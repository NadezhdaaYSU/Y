#!/bin/bash
# gen_acc.sh OUT: the accuracy columns, OUT/<shape>/<column>.ptx, at the shapes of the sweep with fewer than 66
# unsplit CTAs above 32 rows: Y_mma (the unsplit chain, accuracy_auto.py's control), Y_unsplit (the default unsplit
# kernel), Y_auto (`_splitk`, the split-K rule), and Y_s<S>_<tile>, the split the first version of the rule or a
# depth threshold of 2048, or no intensity bound, would take.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; O=$1; mkdir -p $O
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
one() { # shape column name params envs...
  local sh=$1 col=$2 nm=$3 extra=$4 m n k; shift 4; IFS=x read m n k <<< "$sh"
  mkdir -p $O/$sh; cd $O/$sh; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $m $n $k $nm "$extra" > $col.ysu
  env "$@" $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1 || echo "$sh $col: COMPILE FAILED"
  echo "$sh $col: $(grep -m1 -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $col.ptx) | $(grep -m1 -oE 'in [0-9]+ splits?' $col.ptx)"
}
P=', P: GlobalMemory<F32>'
# shape, then the split columns as S:tile
for spec in "512x512x4096 2:64,128,32,4,2,4" "512x512x16384 4:128,128,32,4,2,3" "512x512x65536 4:128,128,32,4,2,3" \
            "256x256x16384 8:64,128,32,4,2,4" "64x4096x4096 2:64,128,32,4,2,4" "64x4096x11008 2:64,128,32,4,2,4" "64x4096x14336 2:64,128,32,4,2,4"; do
  set -- $spec; sh=$1; shift
  one $sh Y_mma g_$sh "" Y_F16_ACCUMULATE=mma
  one $sh Y_unsplit g_$sh "" Y_UNUSED=1
  one $sh Y_auto g_${sh}_splitk "$P" Y_UNUSED=1
  for st in "$@"; do s=${st%%:*}; t=${st#*:}; IFS=, read cm cn ck wm wn sg <<< "$t"
    one $sh Y_s${s}_${cm}x${cn}w${wm}${wn} g_${sh}_splitk$s "$P" Y_CTA_OVERRIDE=$t; done
done
