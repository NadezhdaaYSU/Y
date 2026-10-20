#!/bin/bash
# gen_c3.sh: the stream-K check kernels (labelled K-block variants first, plain last).
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
D=$W/c2; T=128,128,32,4,2,3
for b in 4 8; do Y_STREAMK_KBLOCKS=$b $W/gen_one.sh $D gemm_f16_512x512x4096_streamk66 512 512 4096 4 f32 $T gemm_f16_512x512x4096_streamk66_b$b; done
Y_STREAMK_KBLOCKS=4 $W/gen_one.sh $D gemm_f16_512x512x4096_streamk7 512 512 4096 4 f32 $T gemm_f16_512x512x4096_streamk7_b4
for p in 1 7 66 200; do $W/gen_one.sh $D gemm_f16_512x512x4096_streamk$p 512 512 4096 4 f32 $T; done
$W/gen_one.sh $D gemm_f16_512x512x65536_streamk66 512 512 65536 4 f32 $T
$W/gen_one.sh $D gemm_f16_512x512x4096 512 512 4096 3 f32 $T
