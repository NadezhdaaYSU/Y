#!/bin/bash
# regen_v3.sh: every kernel the queued timing runs read, regenerated with the current compiler into *_new,
# then swapped in for the old (kept as *_v2), all under the GPU lock so no timing run starts in between.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/blk
exec 9>$S/gpu.lock; flock 9
cd $W
rm -rf ptx_new ptx2_new ptx3_new ptx_split_new fused_new
./gen.sh $W/ptx_new > gen_new.log 2>&1
SHAPES="2048x2048x2048 64x4096x4096 1024x4096x4096 128x14336x4096 512x4096x14336 2048x4096x14336" BLOCKS="128 256" ./gen.sh $W/ptx2_new > gen2_new.log 2>&1
SHAPES="$(tr '\n' ' ' < $S/xbar/docs/benchmarks/f16_llm_shapes_2026-10-24/sizes.txt)" BLOCKS="128 256" DEFAULT=1 NO_F32=1 ./gen.sh $W/ptx3_new > gen3_new.log 2>&1
./gen_split.sh $W/ptx_split_new > gen_split_new.log 2>&1
./gen_fused.sh $W/fused_new > gen_fused_new.log 2>&1
for n in 256 1024 4096 8192; do mv ptx_new/${n}x${n}x${n} ptx_new/$n; done; mv ptx2_new/2048x2048x2048 ptx2_new/2048
for sh in 512x512x16384 512x512x65536; do for c in Y_mma Y_blk256; do cp ptx_new/$sh/$c.ptx ptx_split_new/$sh/$c.ptx; done; done
for d in ptx ptx2 ptx3 ptx_split fused; do mv $d ${d}_v2; mv ${d}_new $d; done
mkdir -p fused/blk256cap; sed 's/^\.maxnreg 255/.maxnreg 128/' fused/blk256/gemm_f16_swiglu_4096.ptx > fused/blk256cap/gemm_f16_swiglu_4096.ptx
echo swapped
