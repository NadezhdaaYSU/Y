#!/bin/bash
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$SP/xbar; L=$SP/nr/mut
exec 9>"$SP/gpu.lock"; flock 9
cd $X || exit 9
for m in CONTROL M1_no_tail_predicate M2_tail_not_zeroed M3_weight_offset_by_row M4_newres_from_x M5_guard_wrong_size M6_no_guard M7_old_warp_rule M8_loop_no_newres M9_no_refusal; do
  if [ $m = CONTROL ]; then cat $L/ptx_emitter.rs.orig > src/ptx_emitter.rs; else python3 $L/mut.py $L/ptx_emitter.rs.orig src/ptx_emitter.rs $m || { echo "$m APPLY FAILED" >> $L/summary.txt; continue; }; fi
  touch src/ptx_emitter.rs
  nice -n 19 cargo build --release > $L/$m.build.log 2>&1 || { echo "$m BUILD FAILED" >> $L/summary.txt; continue; }
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=python timeout 1200 python3 -m unittest discover -s python/y_lang/tests -p test_rmsnorm_residual.py > $L/$m.test.log 2>&1
  rc=$?
  echo "$m: exit $rc; $(grep -E '^(FAILED|OK)' $L/$m.test.log | tail -1); failing: $(grep -E '^(FAIL|ERROR):' $L/$m.test.log | sed -E 's/^(FAIL|ERROR): (test_[a-z_0-9]+).*/\2/' | sort | uniq -c | tr -s ' ' | tr '\n' ';')" >> $L/summary.txt
done
cat $L/ptx_emitter.rs.orig > src/ptx_emitter.rs; touch src/ptx_emitter.rs
nice -n 19 cargo build --release > $L/restore.build.log 2>&1
cmp src/ptx_emitter.rs $L/ptx_emitter.rs.orig && echo "RESTORED, build exit $?" >> $L/summary.txt
