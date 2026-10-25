#!/bin/bash
# all_timing.sh: the output-bit check across versions, then every timing batch in turn (each script holds
# the GPU lock and exits 4 if a game is running); stops at the first batch that does not finish.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/blk
cd $W
echo "versions $(date +%T)"; ./run_versions.sh > versions.txt 2>&1; cat versions.txt
echo "core $(date +%T)"; SHAPES="8192 512x512x4096 512x512x16384 512x512x65536" ./run_speed.sh $W/ptx $W/speed || { echo "core exit $?"; exit 1; }
echo "split $(date +%T)"; SHAPES="512x512x16384 512x512x65536" COLS="Y_mma Y_blk256 Y_f32_s4 Y_mma_s4 Y_blk128_s4 Y_blk256_s4" ./run_speed.sh $W/ptx_split $W/split || { echo "split exit $?"; exit 1; }
echo "llm $(date +%T)"; SHAPES="$(cat llm_blocked.txt)" COLS="Y_mma Y_blk128 Y_blk256" ./run_speed.sh $W/ptx3 $W/llm || { echo "llm exit $?"; exit 1; }
echo "fused $(date +%T)"; ./run_fused.sh $W/fused $W/fusedrun || { echo "fused exit $?"; exit 1; }
echo "DONE $(date +%T)"
