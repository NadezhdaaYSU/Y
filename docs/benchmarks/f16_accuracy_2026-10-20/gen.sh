#!/bin/bash
# gen.sh: compile every studied F16 GEMM shape with c67119a's compiler under the
# re-tuned profile, three ways: no knob (default), Y_F16_ACCUMULATE=mma, =f32.
# Kernels land in <mode>/<kernel name>.ptx; default ones are compared with tests/.
set -e
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$S/xbar; D=$S/acc2/ptx
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE
for mode in default mma f32; do
  mkdir -p $D/$mode; cp $S/xbar_ptx/final/.ysu_hw_profile $D/$mode/
  for n in 256 1024 4096 8192; do cp $X/tests/gemm_f16_$n.ysu $D/$mode/; done
  for m in 1 8 32; do cp $X/tests/gemm_f16_decode_m$m.ysu $D/$mode/; done
  for k in 1024 4096 16384 65536; do
    printf '@tile(512, 512, %s)\nkernel gemm_f16_512x512x%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $k $k > $D/$mode/gemm_f16_512x512x$k.ysu
  done
  cd $D/$mode
  for f in *.ysu; do
    b=${f%.ysu}
    if [ $mode = default ]; then env -u Y_F16_ACCUMULATE $X/target/release/Y $f --emit-ptx -o $b.ptx > $b.log 2>&1
    else Y_F16_ACCUMULATE=$mode $X/target/release/Y $f --emit-ptx -o $b.ptx > $b.log 2>&1; fi
  done
  cd - > /dev/null
done
for f in gemm_f16_256 gemm_f16_1024 gemm_f16_4096 gemm_f16_8192 gemm_f16_decode_m1 gemm_f16_decode_m8 gemm_f16_decode_m32; do
  cmp -s $D/default/$f.ptx $X/tests/$f.ptx && echo "$f default == tests/" || echo "$f default DIFFERS from tests/"
done
for f in $D/default/*.ptx; do b=$(basename $f)
  if cmp -s $f $D/mma/$b; then echo "$b default == mma"; elif cmp -s $f $D/f32/$b; then echo "$b default == f32"; else echo "$b default is NEITHER"; fi
done
