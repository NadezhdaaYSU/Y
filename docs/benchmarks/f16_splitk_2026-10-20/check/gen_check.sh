#!/bin/bash
# gen_check.sh OUT_DIR: kernels for check.py, with the compiler in $X.
set -e
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE
mkdir -p $D; cp $S/xbar_ptx/final/.ysu_hw_profile $D/; cd $D
k() { # name m n k [nparams]
  if [ "${5:-3}" = 4 ]; then P=", P: GlobalMemory<F32>"; else P=""; fi
  printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $2 $3 $4 $1 "$P" > $1.ysu
}
k gemm_f16_512x512x4096 512 512 4096; k gemm_f16_512x512x1024 512 512 1024
k gemm_f16_512x512x4096_splitk1 512 512 4096 4; k gemm_f16_512x512x4096_splitk4 512 512 4096 4
k gemm_f16_200x136x4000_splitk4 200 136 4000 4; k gemm_f16_200x136x1000 200 136 1000
k gemm_f16_200x136x4000_splitk1 200 136 4000 4; k gemm_f16_200x136x4000 200 136 4000
for f in *.ysu; do b=${f%.ysu}
  case $b in *splitk*) env -u Y_F16_ACCUMULATE $X/target/release/Y $f --emit-ptx -o $b.ptx > $b.log 2>&1 ;;
             *) Y_F16_ACCUMULATE=f32 $X/target/release/Y $f --emit-ptx -o $b.ptx > $b.log 2>&1 ;; esac
done
mkdir -p mma; cp gemm_f16_512x512x4096_splitk4.ysu .ysu_hw_profile mma/
(cd mma && Y_F16_ACCUMULATE=mma $X/target/release/Y gemm_f16_512x512x4096_splitk4.ysu --emit-ptx -o gemm_f16_512x512x4096_splitk4.ptx > log 2>&1)
grep -H "Y TENSOR CORE GEMM\]" *.ptx mma/*.ptx | sed 's/ *\/\/ \[Y TENSOR CORE GEMM\]//'
