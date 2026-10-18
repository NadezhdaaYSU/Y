#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
for i in 1 2; do WAIT_S=${WAIT_S:-14400} "$SP/gpu_quiet_then.sh" bash "$SP/f16sq/run.sh" $i; done
