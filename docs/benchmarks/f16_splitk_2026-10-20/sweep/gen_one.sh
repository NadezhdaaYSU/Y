#!/bin/bash
# gen_one.sh DIR NAME M N K NPARAMS MODE TILE: compile one @tile kernel into DIR/NAME.ptx
# (MODE f32|mma|default, TILE Y_CTA_OVERRIDE or auto) and print its header and ptxas
# registers; with a 9th argument the files are renamed to that label.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=${X:-$S/xbar}; D=$1; nm=$2
unset Y_CTA_OVERRIDE Y_SMEM_SWIZZLE Y_SMEM_PAD Y_SWIGLU_TILE Y_F16_ACCUMULATE
mkdir -p $D; cd $D; [ -f .ysu_hw_profile ] || cp $S/xbar_ptx/final/.ysu_hw_profile .
if [ $6 = 4 ]; then P=", P: GlobalMemory<F32>"; else P=""; fi
printf '@tile(%s, %s, %s)\nkernel %s(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>%s) {\n}\n\nfn main() {\n}\n' $3 $4 $5 $nm "$P" > $nm.ysu
envs=(); [ "$7" != default ] && envs+=(Y_F16_ACCUMULATE=$7); [ "$8" != auto ] && envs+=(Y_CTA_OVERRIDE=$8)
env "${envs[@]}" $X/target/release/Y $nm.ysu --emit-ptx -o $nm.ptx > $nm.log 2>&1; rc=$?
if [ $rc != 0 ] || [ ! -s $nm.ptx ]; then echo "$nm: compile exit $rc: $(grep -iE 'error|refus' $nm.log | head -3)"; exit 0; fi
echo "$nm: $(grep -oE '(STREAM|SPLIT)-K GEMM\] [^|]*\| grid \([0-9, ]*\)' $nm.ptx | head -1) | $(grep -oE 'CTA [0-9x]+ \| [0-9x]+ warps' $nm.ptx | head -1) | $(ptxas -arch=sm_89 -v $nm.ptx -o /dev/null 2>&1 | grep -A3 "function '$nm'" | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' ')"
[ -n "$9" ] && mv $nm.ptx $9.ptx && mv $nm.log $9.log
