#!/bin/bash
# run.sh: each mutant of the swizzled GEMM tiles, rebuilt, against
# tests/ptx_smem_swizzle.rs; the unmutated source as the control; then restore.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$SP/xbar; L=$SP/swz/mut
exec 9>"$SP/gpu.lock"; flock 9
cd $X || exit 9
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs || { echo "SOURCE CHANGED SINCE THE COPY" >> $L/summary.txt; exit 8; }
for m in CONTROL M1_copies_unswizzled M2_a_step_added_not_xored M3_b_block_ored_not_xored M4_mask_full_width M5_shift_zero M6_no_period_check M7_lane_half_dropped M8_sync_copy_unswizzled M9_chunk_copy_unswizzled M10_hoisted_copy_unswizzled M11_padded_by_default M12_b_lane_term_from_a M13_a_lane_term_from_b; do
  if [ $m = CONTROL ]; then cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; else python3 $L/mut.py $L/ptx_emitter.final.rs src/ptx_emitter.rs $m || { echo "$m APPLY FAILED" >> $L/summary.txt; continue; }; fi
  touch src/ptx_emitter.rs
  nice -n 19 cargo test --release --test ptx_smem_swizzle --no-run > $L/$m.build.log 2>&1 || { echo "$m BUILD FAILED" >> $L/summary.txt; continue; }
  timeout 1200 nice -n 19 cargo test --release --test ptx_smem_swizzle > $L/$m.test.log 2>&1
  rc=$?
  echo "$m: exit $rc; $(grep -E '^test result' $L/$m.test.log | tail -1); failing: $(grep -E '^test .* FAILED' $L/$m.test.log | sed -E 's/^test ([a-z_0-9]+) .*/\1/' | tr '\n' ' ')" >> $L/summary.txt
done
cat $L/ptx_emitter.final.rs > src/ptx_emitter.rs; touch src/ptx_emitter.rs
nice -n 19 cargo build --release > $L/restore.build.log 2>&1; b=$?
cmp src/ptx_emitter.rs $L/ptx_emitter.final.rs && echo "RESTORED, build exit $b" >> $L/summary.txt
