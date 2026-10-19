#!/bin/bash
# run.sh: each mutant of the constant-division pass, rebuilt, against ptx_constant_division.rs; then restore.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$SP/xbar; L=$SP/cdiv/mut
exec 9>"$SP/gpu.lock"; flock 9
cd $X || exit 9
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs || { echo "SOURCE CHANGED SINCE THE COPY" >> $L/summary.txt; exit 8; }
for m in CONTROL M1_shift_off_by_one M2_mask_is_c M3_no_upper_bound M4_rem_aliasing_allowed M5_guard_on_first_only M6_signed_too M7_pass_not_applied M8_u64_mask_b32 M9_div_by_one_is_zero M10_epilogue_unrolled; do
  if [ $m = CONTROL ]; then cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; else python3 $L/mut.py $L/ptx_emitter.final.rs src/ptx_emitter.rs $m || { echo "$m APPLY FAILED" >> $L/summary.txt; continue; }; fi
  touch src/ptx_emitter.rs
  nice -n 19 cargo test --release --test ptx_constant_division --no-run > $L/$m.build.log 2>&1 || { echo "$m BUILD FAILED" >> $L/summary.txt; continue; }
  timeout 1200 nice -n 19 cargo test --release --test ptx_constant_division > $L/$m.test.log 2>&1
  rc=$?
  echo "$m: exit $rc; $(grep -E '^test result' $L/$m.test.log | tail -1); failing: $(grep -E '^test .* FAILED' $L/$m.test.log | sed -E 's/^test ([a-z_0-9]+) .*/\1/' | tr '\n' ' ')" >> $L/summary.txt
done
cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; touch src/ptx_emitter.rs
nice -n 19 cargo build --release > $L/restore.build.log 2>&1; b=$?
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs && echo "RESTORED, build exit $b" >> $L/summary.txt
