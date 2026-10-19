#!/bin/bash
# census.sh <old tree-ish> <new dir> <out>: static SASS of every kernel the pass changed, old and new:
# total instructions, I2F (one per reciprocal-division setup), F2I, IMAD.HI and the dead @!PT LDS.
OLD=$1; NEW=$2; OUT=$3; T=$(mktemp -d)
printf "%-32s %14s %14s %14s %14s\n" kernel "SASS old>new" "I2F old>new" "IMAD.HI" "@!PT LDS" > $OUT
for f in $(git diff --name-only $OLD -- 'tests/*.ptx'); do
  k=$(basename $f .ptx)
  git show $OLD:$f > $T/o.ptx; cp $NEW/$k.ptx $T/n.ptx
  for v in o n; do ptxas -arch=sm_89 -o $T/$v.cubin $T/$v.ptx && nvdisasm -c $T/$v.cubin > $T/$v.sass; done
  c() { grep -cE "$1" $T/$2.sass; }
  printf "%-32s %6s>%-7s %6s>%-7s %6s>%-7s %6s>%-7s\n" $k $(c '/\*[0-9a-f]{4}\*/' o) $(c '/\*[0-9a-f]{4}\*/' n) $(c 'I2F' o) $(c 'I2F' n) $(c 'IMAD.HI' o) $(c 'IMAD.HI' n) $(c '@!PT LDS' o) $(c '@!PT LDS' n) >> $OUT
done
rm -rf $T
