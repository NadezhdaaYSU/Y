#!/bin/bash
# Paged decode attention record, 2026-10-18: waits for a quiet GPU (no game,
# 10-sample mean <= 5%) holding gpu.lock, then
#   1. cupti_attn.py: device times of the old (7e08614) and new kernels and
#      FlashInfer, interleaved in one process, all six README cases;
#   2. the README's benchmark script with the old compiler, then the new.
# Everything runs from snapshots (snap_old, snap_new) of the compilers, PTX
# and scripts, so later edits to a worktree cannot change what is measured.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
B=$SP/bench_attn
export PYTHONPATH=$SP/fishim/y_flashinfer_shim
  {
  date; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader
  sha256sum $SP/snap_old/target/release/Y $SP/snap_new/target/release/Y
  python3 -c "import torch, flashinfer, cupy; print('torch', torch.__version__, 'flashinfer', flashinfer.__version__, 'cupy', cupy.__version__)"
} > $B/env_2.txt 2>&1
cd $B/ptx || exit 9
Y_TESTS=$SP/snap_new/tests python3 $B/cupti_attn.py \
    16:8:old_16_8.ptx 4:8:old_4_8.ptx 16:8:new_16_8.ptx 4:8:new_4_8.ptx 4:4:new_4_4.ptx 32:4:new_32_4.ptx \
    --rounds 7 2>&1 | grep -v USDT > $B/cupti_2.log
echo "cupti exit ${PIPESTATUS[0]}" >> $B/summary_2.txt
for t in old new; do
  cd $SP/snap_$t || exit 9
  python3 tests/benchmark_y_paged_decode_attention.py > $B/readme_bench_${t}_2.log 2>&1
  echo "readme bench $t exit $?" >> $B/summary_2.txt
done
nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader >> $B/env_2.txt
"$SP/games.sh" >> $B/env_2.txt
date >> $B/summary_2.txt
