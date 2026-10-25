#!/bin/bash
# diag/gen.sh: at three shapes, the chain, blocked 256, and blocked with one block of all K (one fold, after the
# K loop): what blocking costs apart from its folds.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/blk
X=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar
cd $W/diag
for t in "8192 8192 8192 8192 ptx/8192" "512x512x65536 512 512 65536 ptx/512x512x65536" "512x4096x4096 512 4096 4096 ptx3/512x4096x4096"; do
  set -- $t
  mkdir -p $1; cp $W/ptx/8192/.ysu_hw_profile $1/; cp $W/$5/Y_mma.ptx $W/$5/Y_blk256.ptx $1/
  printf '@tile(%s, %s, %s)\nkernel gemm_f16_%sx%sx%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $2 $3 $4 $2 $3 $4 > $1/Y_blkK.ysu
  (cd $1 && Y_F16_ACCUMULATE=blocked Y_F16_BLOCK_K=$4 $X/target/release/Y Y_blkK.ysu --emit-ptx -o Y_blkK.ptx > Y_blkK.log 2>&1)
  echo "$1: inner-loop labels $(grep -c 'F16_BLOCK_TILE' $1/Y_blkK.ptx), tile $(grep -m1 -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $1/Y_blkK.ptx)"
done
