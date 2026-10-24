#!/bin/bash
# sweep_tiles.sh: the four lagging shapes under eight tiles, one directory per tile.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
SH="128x11008x4096 256x11008x4096 256x4096x11008 1024x11008x4096"
for t in 64,64,64,2,2,4 64,128,32,2,4,3 64,128,32,2,2,3 128,64,32,2,2,3 128,64,32,4,2,3 128,128,32,2,2,3 128,128,32,4,2,3 64,64,32,2,2,4; do
  $W/llm/gen_tile.sh $W/llm/t_${t//,/_} $t $SH
done
