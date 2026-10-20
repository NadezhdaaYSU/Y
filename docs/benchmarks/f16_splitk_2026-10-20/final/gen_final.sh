#!/bin/bash
# gen_final.sh OUT: the kernels of the final speed table, with the compiler in $X.
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
O=$1
g() { # K label splits mode tile
  local nm=gemm_f16_512x512x$1 np=3
  [ $3 != 0 ] && nm=${nm}_splitk$3 && np=4
  $W/gen_one.sh $O/$1 $nm 512 512 $1 $np $4 $5 $2 | cut -c1-160
}
g 4096 default 0 default auto; g 4096 f32 0 f32 auto
g 4096 s2_f32 2 f32 64,64,32,4,2,4; g 4096 s4_f32 4 f32 64,128,32,4,2,4; g 4096 s2_mma 2 mma 64,64,32,4,2,4
g 16384 default 0 default auto; g 16384 f32 0 f32 auto
g 16384 s2_f32 2 f32 64,64,32,4,2,4; g 16384 s4_f32 4 f32 128,128,32,4,2,3; g 16384 s8_f32 8 f32 128,128,32,4,2,3; g 16384 s4_mma 4 mma 128,128,32,4,2,3
g 65536 default 0 default auto; g 65536 f32 0 f32 auto
g 65536 s4_f32 4 f32 128,128,32,4,2,3; g 65536 s8_f32 8 f32 128,128,32,4,2,3; g 65536 s16_f32 16 f32 128,128,32,4,2,3; g 65536 s4_mma 4 mma 128,128,32,4,2,3
