#!/bin/bash
# census.sh OUT: for each shape, the tile the profile gives the unsplit kernel and each split S in 2/4/8,
# with grid CTAs, registers, smem, and the accumulation mode the header names.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; D=$1; mkdir -p $D
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE Y_F16_BLOCK_K
cp $S/xbar_ptx/final/.ysu_hw_profile $D/
cd $D
SHAPES="512x512x4096 512x512x16384 512x512x65536 256x256x16384 1024x1024x16384 $(cat "$X/docs/benchmarks/f16_llm_shapes_2026-10-24/sizes.txt") 1024x1024x1024 2048x2048x2048 4096x4096x4096 8192x8192x8192"
for M in 1 8 32; do for N in 4096 14336; do for K in 4096 14336; do SHAPES="$SHAPES ${M}x${N}x${K}"; done; done; done
for sh in $SHAPES; do
  IFS=x read m n k <<< "$sh"
  for s in 1 2 4 8; do
    if [ $s = 1 ]; then name=g_$sh; sig="C: GlobalMemory<F32>"; else
      [ $((k % s)) = 0 ] && [ $(((k / s) % 8)) = 0 ] || continue
      name=g_${sh}_splitk$s; sig="C: GlobalMemory<F32>, P: GlobalMemory<F32>"; fi
    f=${sh}_s$s
    printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, %s) {\n}\n\nfn main() {\n}\n' $m $n $k $name "$sig" > $f.ysu
    $X/target/release/Y $f.ysu --emit-ptx -o $f.ptx > $f.log 2>&1 || { echo "$sh s$s COMPILE FAILED"; continue; }
    tile=$(grep -m1 -oE 'CTA [0-9]+x[0-9]+x[0-9]+ \| [0-9]+x[0-9]+ warps' $f.ptx)
    smem=$(grep -m1 -oE 'Dynamic shared memory required: [0-9]+' $f.ptx | grep -oE '[0-9]+$')
    src=$(grep -m1 -oE 'measured tile from|borrowing the measured tile from the nearest shape M=[0-9]+ N=[0-9]+ K=[0-9]+|analytic' $f.log | head -1)
    regs=$(ptxas -arch=sm_89 -v $f.ptx -o /dev/null 2>&1 | grep -A1 "for '$name'" | grep -oE 'Used [0-9]+ registers' | grep -oE '[0-9]+')
    spill=$(ptxas -arch=sm_89 -v $f.ptx -o /dev/null 2>&1 | grep -A1 "for '$name'" | grep -oE '[0-9]+ bytes spill stores' | grep -oE '^[0-9]+')
    blk=$(grep -c F16_BLOCK_KEEP $f.ptx); f32=$(grep -cE '\{(%f[0-9]+),\1,\1,\1\}' $f.ptx)
    echo "$sh s$s | $tile | smem $smem | regs $regs spill ${spill:-0} | blockmarks $blk f32zero $f32 | $src"
  done
done
