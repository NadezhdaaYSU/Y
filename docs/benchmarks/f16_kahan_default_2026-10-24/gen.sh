#!/bin/bash
# gen.sh OUT: decode GEMMs (g<MxNxK>.ptx, kernel gemm_f16_<MxNxK>) at N or K = 11008 / 14336,
# in OUT/<mode> for the knob unset
# (default) and Y_F16_ACCUMULATE=f32, kahan and mma, by the compiler in $X with the re-tuned
# profile the committed PTX came from; tile, registers and spills of each.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
GEMMS="1x11008x4096 8x11008x4096 32x11008x4096 1x4096x11008 8x4096x11008 32x4096x11008 1x14336x4096 8x14336x4096 32x14336x4096 1x4096x14336 8x4096x14336 32x4096x14336"
regs() { ptxas -arch=sm_89 -v $1 -o /dev/null 2>&1 | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' '; }
for mode in default f32 kahan mma; do
  mkdir -p $D/$mode; cd $D/$mode || exit 1; cp $S/xbar_ptx/final/.ysu_hw_profile .
  knob=(); [ $mode != default ] && knob=(env Y_F16_ACCUMULATE=$mode)
  for size in $GEMMS; do
    IFS=x read m n k <<< "$size"
    printf '@tile(%s, %s, %s)\nkernel gemm_f16_%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $size > g$size.ysu
    "${knob[@]}" $X/target/release/Y g$size.ysu --emit-ptx -o g$size.ptx > g$size.log 2>&1 || echo "$mode $size: compile failed"
    echo "$mode gemm $size: $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' g$size.ptx | head -1) | $(regs g$size.ptx)"
  done
done
# The default must be the compensated kernel at every one of these shapes (all M <= 32).
cd $D; n=0; for p in default/*.ptx; do cmp -s $p kahan/$(basename $p) || { echo "default differs from kahan: $p"; n=$((n+1)); }; done; echo "default == kahan: $(ls default/*.ptx | wc -l) files, $n differ"
