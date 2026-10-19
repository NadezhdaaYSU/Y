#!/bin/bash
# make_variants.sh <out dir>: regenerates the 141 hand-written mappings the
# discovery runs timed (v<hd>/v_<tpr>_<slots>_<unroll>.ptx), each assembled
# by ptxas; variants.sha256 lists what was measured.
D=$(cd "$(dirname "$0")" && pwd); O=${1:?out dir}
for hd in 64 128 256; do
  mkdir -p "$O/v$hd"
  for tpr in 128 64 32 16 8; do
    [ $tpr -le $((hd / 2)) ] || continue
    for th in 64 128 256 512; do
      [ $th -ge $tpr ] || continue
      for u in 1 2 4; do
        f="$O/v$hd/v_${tpr}_$((th / tpr))_${u}.ptx"
        python3 "$D/mk_rope.py" "$f" $hd $tpr $((th / tpr)) $u && ptxas -arch=sm_89 "$f" -o /dev/null || exit 1
      done
    done
  done
done
