#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
K=$1; shift
for spec in "$@"; do IFS=';' read P B mode <<< "$spec"
  Y_STREAMK_KBLOCKS=$B $W/gen_one.sh $W/sw$K gemm_f16_512x512x${K}_streamk$P 512 512 $K 4 $mode 128,128,32,4,2,3 sk_p${P}_b${B}_$mode
done
