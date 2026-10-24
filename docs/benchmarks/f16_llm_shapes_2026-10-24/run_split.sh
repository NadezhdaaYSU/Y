#!/bin/bash
# run_split.sh: sbench.py on every kernel of split/<M>, two runs, small-integer and N(0,1) data.
S=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
W=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8cb989c6-e604-4dd6-849f-ecfdae469440/scratchpad/sk
export Y_LIB_PATH=$S/xbar/target/release/liby.so PYTHONPATH=$S/xbar/python
for i in 1 2; do for m in 128 256; do for d in int normal; do
  cols=""; for f in $W/llm/split/$m/*.ptx; do cols="$cols $(basename $f .ptx)=$f"; done
  python3 $S/xbar/docs/benchmarks/f16_splitk_2026-10-20/sweep/sbench.py $W/llm/split/sb_${m}_${i}_$d.json $m 11008 4096 $cols calls=20 data=$d > $W/llm/split/sb_${m}_${i}_$d.log 2>&1
  echo "m $m run $i $d exit $?"
done; done; done
