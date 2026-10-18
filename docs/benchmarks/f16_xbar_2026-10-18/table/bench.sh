#!/bin/bash
# One bench_ab.py run of the six square F16 GEMMs, compiled with the K-loop
# schedule change from tests/gemm_f16_<N>.ysu against the main checkout's
# profile (byte-identical to the regenerated tests/gemm_f16_<N>.ptx), against
# Triton and cuBLAS. Started by gpu_quiet_then.sh (holds gpu.lock; mean
# utilisation <= 5%, no game).
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
T=$SP/xbar_ptx/table
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } > $T/env_$1.txt
cd $T || exit 9
Y_LIB_PATH=$SP/snap957/target/release/liby.so PYTHONPATH=$SP/snap957/python python3 $SP/f16sq/bench_ab.py "Y=$T/y" $T/run$1.json 256,512,1024,2048,4096,8192 > $T/run$1.log 2>&1
echo "run $1 exit $?" >> $T/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } >> $T/env_$1.txt
