#!/bin/bash
# Waits for a quiet GPU (no game, 10-sample mean <= 5%) holding gpu.lock, then runs run_all.sh.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
WAIT_S=${WAIT_S:-14400} "$SP/gpu_quiet_then.sh" bash "$SP/bench_attn/run_all.sh"
