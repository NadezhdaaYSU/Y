#!/bin/bash
# run_balance.sh: does a split near the DRAM balance want two CTAs an SM? Shapes with fewer than 66 unsplit CTAs
# and 96-200 flops a byte, each S of 2, 4, 8 on 64x64 and 64x128 (4x2 warps), timed with A and B in the L2 and
# from DRAM, with the main sweep's harness. Prepared 2026-10-25 while the GPU was busy; not run.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
D=$S/xbar/docs/benchmarks/f16_auto_splitk_2026-10-25; O=$S/auto_balance_ptx; F=$(dirname $0)
SH="256x256x16384 256x256x32768 256x256x65536 256x512x16384 256x384x32768"
rm -rf $O; mkdir -p $O
for sh in $SH; do IFS=x read m n k <<< "$sh"; mkdir -p $O/$sh; cd $O/$sh; cp $S/xbar_ptx/final/.ysu_hw_profile .
  printf '@tile(%s, %s, %s)\nkernel g(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k > unsplit.ysu
  $S/xbar/target/release/Y unsplit.ysu --emit-ptx -o unsplit.ptx > unsplit.log 2>&1
  printf '@tile(%s, %s, %s)\nkernel g_splitk(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, P: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k > auto.ysu
  $S/xbar/target/release/Y auto.ysu --emit-ptx -o auto.ptx > auto.log 2>&1
  for s in 2 4 8; do for t in 64,64,32,4,2,4 64,128,32,4,2,4; do IFS=, read cm cn ck wm wn st <<< "$t"
    printf '@tile(%s, %s, %s)\nkernel g_splitk%s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, P: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $s > s.ysu
    Y_CTA_OVERRIDE=$t $S/xbar/target/release/Y s.ysu --emit-ptx -o s${s}_${cm}x${cn}.ptx > s${s}_${cm}x${cn}.log 2>&1; done; done
done
$D/run_speed.sh $O $F/l2 "$SH" && DRAM=1 $D/run_speed.sh $O $F/dram "$SH"
for r in l2 dram; do echo "== $r"; python3 $D/columns.py $F/$r $SH; done > $F/summary.txt
