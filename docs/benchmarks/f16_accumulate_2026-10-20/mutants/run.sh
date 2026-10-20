#!/bin/bash
# run.sh: each mutant of the F16 float32 accumulation, rebuilt, against tests/ptx_f16_accumulate.rs; the source as control; restore.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$SP/xbar; L=$SP/f32acc/mut
exec 9>"$SP/gpu.lock"; flock 9
cd $X || exit 9
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs || { echo "SOURCE CHANGED" >> $L/summary.txt; exit 8; }
for m in CONTROL M1_mma_from_accumulator M2_no_adds M3_threshold_below_32 M4_f32_knob_ignored M5_bad_knob_accepted M6_round_toward_zero; do
  if [ $m = CONTROL ]; then cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; else python3 $L/mut.py $L/ptx_emitter.final.rs src/ptx_emitter.rs $m || { echo "$m APPLY FAILED" >> $L/summary.txt; continue; }; fi
  touch src/ptx_emitter.rs
  nice -n 19 cargo test --release --test ptx_f16_accumulate > $L/$m.test.log 2>&1
  echo "$m: exit $?; $(grep -E '^test result' $L/$m.test.log | tail -1 | cut -c1-40); failing: $(grep -E '^test .* FAILED' $L/$m.test.log | sed -E 's/^test ([a-z_0-9]+) .*/\1/' | tr '\n' ' ')| $(grep -h 'panicked' -A1 $L/$m.test.log | grep -v panicked | head -1 | cut -c1-100)" >> $L/summary.txt
done
cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; touch src/ptx_emitter.rs
nice -n 19 cargo build --release > $L/restore.build.log 2>&1; b=$?
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs && echo "RESTORED, build exit $b" >> $L/summary.txt
