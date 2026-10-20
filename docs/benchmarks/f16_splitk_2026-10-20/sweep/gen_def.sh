#!/bin/bash
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
for K in 4096 16384 65536; do for s in 2 4 8; do
  $W/gen_one.sh $W/sw$K gemm_f16_512x512x${K}_splitk$s 512 512 $K 4 default auto def_s${s} | cut -c1-200
  grep -aE "no measurement|borrowing|cached|measured" $W/sw$K/def_s${s}.log | head -2 | cut -c1-200
done; done
