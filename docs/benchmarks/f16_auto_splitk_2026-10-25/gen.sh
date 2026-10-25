#!/bin/bash
# gen.sh OUT [SHAPES]: every column of every shape into OUT/<shape>/<column>.ptx, with the compiler in $X and
# the profile $S/xbar_ptx/final/.ysu_hw_profile:
#   unsplit   - the plain kernel (the profile's tile, the default sum);
#   auto      - `<name>_splitk`, the split-K rule's choice;
#   s<S>_<t>  - `<name>_splitk<S>` pinned to tile t (Y_CTA_OVERRIDE), for S in 2, 4, 8 with K/S >= 1024:
#               above 32 rows each of the rule's tiles that fits M x N (128x128 4x2 s3, 64x128 2x4 s4,
#               64x64 4x2 s4), at the A shapes also 64x128 4x2 s4 and 128x128 4x2 s4; at 32 rows or fewer
#               the unsplit kernel's tile. At the B shapes (shapes.sh) only the two largest tiles that fit, and
#               no split whose workspace (S x M x N floats) passes 1 GiB.
# Writes OUT/<shape>/<column>.log and one line per column to OUT/columns.txt.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; O=$1; mkdir -p $O
. $X/docs/benchmarks/f16_auto_splitk_2026-10-25/shapes.sh
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
one() { # shape column name params tile
  local sh=$1 col=$2 nm=$3 extra=$4 tile=$5 m n k; IFS=x read m n k <<< "$sh"
  mkdir -p $O/$sh; cd $O/$sh; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $m $n $k $nm "$extra" > $col.ysu
  if [ -n "$tile" ]; then env Y_CTA_OVERRIDE=$tile $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1
  else $X/target/release/Y $col.ysu --emit-ptx -o $col.ptx > $col.log 2>&1; fi || { echo "$sh $col: COMPILE FAILED" | tee -a $O/columns.txt; return; }
  echo "$sh $col: $(grep -m1 -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $col.ptx) | $(grep -m1 -oE 'in [0-9]+ splits?' $col.ptx) | $(ptxas -arch=sm_89 -v $col.ptx -o /dev/null 2>&1 | grep -A1 "for '$nm'" | grep -oE 'Used [0-9]+ registers' ; ptxas -arch=sm_89 -v $col.ptx -o /dev/null 2>&1 | grep -A2 "for '$nm'" | grep -oE '[0-9]+ bytes spill stores') | $(grep -m1 -oE 'S=[0-9]+ by the split-K rule: .*' $col.ptx)" >> $O/columns.txt
}
for sh in ${2:-$SHAPES_A $SHAPES_D $SHAPES_B}; do
  IFS=x read m n k <<< "$sh"; P=', P: GlobalMemory<F32>'
  one $sh unsplit g_$sh "" ""
  one $sh auto g_${sh}_splitk "$P" ""
  if [ $m -gt 32 ]; then
    tiles="128,128,32,4,2,3 64,128,32,2,4,4 64,64,32,4,2,4"
    case " $SHAPES_A " in *" $sh "*) tiles="$tiles 64,128,32,4,2,4 128,128,32,4,2,4";; esac
  else
    t=$(grep -m1 -oE 'CTA Tile: [0-9]+x[0-9]+x[0-9]+, Warps: [0-9]+, Pipeline Stages: [0-9]+' $O/$sh/unsplit.log)
    read cm cn ck <<< "$(echo $t | grep -oE '[0-9]+x[0-9]+x[0-9]+' | tr x ' ')"
    wm=$(grep -m1 -oE "CTA ${cm}x${cn}x${ck} \| [0-9]+x[0-9]+ warps" $O/$sh/unsplit.ptx | grep -oE '\| [0-9]+x' | grep -oE '[0-9]+')
    wn=$(grep -m1 -oE "CTA ${cm}x${cn}x${ck} \| [0-9]+x[0-9]+ warps" $O/$sh/unsplit.ptx | grep -oE 'x[0-9]+ warps' | grep -oE '[0-9]+')
    st=$(echo $t | grep -oE 'Stages: [0-9]+' | grep -oE '[0-9]+')
    tiles="$cm,$cn,$ck,$wm,$wn,$st"
  fi
  for s in 2 4 8; do
    [ $((k % s)) = 0 ] && [ $(((k / s) % 8)) = 0 ] && [ $((k / s)) -ge 1024 ] || continue
    case " $SHAPES_B " in *" $sh "*) b=1;; *) b=0;; esac
    [ $b = 1 ] && [ $((s * m * n * 4)) -gt 1073741824 ] && continue
    used=0
    for t in $tiles; do
      IFS=, read cm cn ck wm wn st <<< "$t"; [ $m -gt 32 ] && { [ $cm -le $m ] && [ $cn -le $n ] || continue; }
      [ $b = 1 ] && [ $used = 2 ] && continue
      used=$((used + 1))
      one $sh s${s}_${cm}x${cn}w${wm}${wn}s${st} g_${sh}_splitk$s "$P" $t
    done
  done
done
