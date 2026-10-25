#!/bin/bash
# run_acc.sh PTX OUT.json: accuracy_auto.py on every shape directory of PTX (gen_acc.sh), under the GPU lock;
# the environment (GPU, utilisation, games, compiler commit and diff) in OUT.env. Accuracy does not depend on
# contention, so a game does not stop it; it is recorded.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; P=$1; O=$2
export ACC_DIR=$X/docs/benchmarks/f16_accuracy_2026-10-20 SPLIT_DIR=$X/docs/benchmarks/f16_splitk_2026-10-20 Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=$X/python
exec 9>$S/gpu.lock; flock 9
(date; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm,power.draw --format=csv,noheader; $S/games.sh; echo; git -C $X rev-parse HEAD; git -C $X diff --stat) > ${O%.json}.env
python3 $X/docs/benchmarks/f16_auto_splitk_2026-10-25/accuracy/accuracy_auto.py $X/docs/benchmarks/f16_square_2026-10-18 $P $O > ${O%.json}.log 2>&1
echo "exit $?" >> ${O%.json}.log
