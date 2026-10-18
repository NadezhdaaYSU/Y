#!/bin/bash
# One bench_ab.py run of the six square F16 GEMMs (compiled at bc0bbc3 from
# tests/gemm_f16_<N>.ysu against the main checkout's profile; byte-identical
# to the committed tests/gemm_f16_<N>.ptx) against Triton and cuBLAS.
# Started by gpu_quiet_then.sh (holds gpu.lock; mean utilisation <= 5%, no game).
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
F=$SP/f16sq
{ date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } > $F/env_$1.txt
cd $F || exit 9
Y_LIB_PATH=$F/snap/target/release/liby.so PYTHONPATH=$F/snap/python python3 $F/bench_ab.py "Y=$F/ptx" $F/run$1.json 256,512,1024,2048,4096,8192 > $F/run$1.log 2>&1
echo "run $1 exit $?" >> $F/summary.txt
{ date; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader; "$SP/games.sh"; echo; } >> $F/env_$1.txt
