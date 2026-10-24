#!/bin/bash
# gen_split.sh OUT: 128 and 256 x 11008 x 4096, unsplit and split 2/3/4, float32 (default) and chained, two tiles.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
SK=$S/xbar/docs/benchmarks/f16_splitk_2026-10-20/sweep/gen_one.sh
O=$1
for m in 128 256; do for tile in 64,64,32,2,2,4 64,128,32,2,2,3; do t=${tile//,/_}
  $SK $O/$m gemm_f16_${m}x11008x4096 $m 11008 4096 3 default $tile u_$t | cut -c1-120
  for s in 2 4; do for mode in f32 mma; do
    $SK $O/$m gemm_f16_${m}x11008x4096_splitk$s $m 11008 4096 4 $mode $tile s${s}_${mode}_$t | cut -c1-120
  done; done
done; done
