#!/bin/bash
# run.sh: Y --autotune-force on every committed F16 GEMM shape, into a copy of the
# profile the committed kernels were generated under; under gpu.lock.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
T=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/autotune2
X=$SP/xbar
exec 9>"$SP/gpu.lock"; flock 9
cd $T || exit 9
echo "start $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) games: $("$SP/games.sh" | tr "\n" " ")" >> $T/summary.txt
for k in gemm_f16_256 gemm_f16_512 gemm_f16_1024 gemm_f16_2048 gemm_f16_4096 gemm_f16_8192 gemm_f16_16384 gemm_f16_decode_m1 gemm_f16_decode_m4 gemm_f16_decode_m8 gemm_f16_decode_m32 tensor_core_gemm_f16_rect gemm_f16_uneven_grid; do
  cp $X/tests/$k.ysu $T/
  s=$(date +%s); timeout 3600 $X/target/release/Y $k.ysu --emit-ptx --autotune-force -o $k.ptx > $k.log 2>&1
  echo "$k exit $? $(( $(date +%s)-s ))s: $(grep -o "JIT Autotuner Result.*" $k.log | head -1)" >> $T/summary.txt
done
diff <(grep AUTOTUNE $T/profile.before) <(grep AUTOTUNE $T/.ysu_hw_profile) > $T/profile.diff
echo "DONE $(date +%T)" >> $T/summary.txt
