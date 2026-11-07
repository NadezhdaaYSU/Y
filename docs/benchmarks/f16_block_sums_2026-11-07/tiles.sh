#!/usr/bin/env bash
# tiles.sh PROFILE ELEM M N K ... : print the tile and block-sum add count Y picks.
set -u
Y="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/target/release/Y"
prof=$1; elem=$2; shift 2
while [ $# -ge 3 ]; do
  m=$1 n=$2 k=$3; shift 3
  d=$(mktemp -d); cp "$prof" "$d/.ysu_hw_profile"
  printf '@tile(%s, %s, %s)\nkernel g(A: GlobalMemory<%s>, B: GlobalMemory<%s>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n' $m $n $k $elem $elem > "$d/g.ysu"
  (cd "$d" && "$Y" g.ysu --emit-ptx -o g.ptx > log.txt 2>&1)
  if [ -f "$d/g.ptx" ]; then
    echo "${m}x${n}x${k} $elem: $(grep -o 'CTA [0-9x]* | [0-9x]* warps' "$d/g.ptx") s=$(grep -o '[0-9]* used after' "$d/g.ptx" | cut -d' ' -f1) blockadds=$(grep -c 'add.f32\|add.rn.f32' "$d/g.ptx")"
  else
    echo "${m}x${n}x${k}: FAILED"; tail -3 "$d/log.txt"
  fi
  rm -rf "$d"
done
