#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
for K in 4096 16384 65536; do for P in 66 132; do for mode in f32 mma; do
  $W/gen_one.sh $W/sw$K gemm_f16_512x512x${K}_streamk$P 512 512 $K 4 $mode 128,128,32,4,2,3 sk_128_128_32_4_2_3_p${P}_$mode
done; done; done
