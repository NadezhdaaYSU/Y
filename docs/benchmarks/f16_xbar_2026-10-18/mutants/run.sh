#!/bin/bash
# Each mutant: write, touch, build, run test_f16_gemm.py; then restore by writing, touch, rebuild.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
X=$SP/xbar; L=$SP/xbar_mut
exec 9>"$SP/gpu.lock"; flock 9
cd $X || exit 9
for m in CONTROL M1_wait_one_fewer M2_no_mid_barrier M3_loads_above_wait M4_next_from_k_iter M5_no_commit_before_wait M6_two_stages_too M7_no_first_fragments; do
  if [ $m = CONTROL ]; then cat $L/ptx_emitter.rs.orig > src/ptx_emitter.rs; else python3 $L/mut.py $L/ptx_emitter.rs.orig src/ptx_emitter.rs $m || { echo "$m APPLY FAILED" >> $L/summary.txt; continue; }; fi
  touch src/ptx_emitter.rs
  nice -n 19 cargo build --release > $L/$m.build.log 2>&1 || { echo "$m BUILD FAILED" >> $L/summary.txt; continue; }
  (mkdir -p $L/ptx_$m && cd $L/ptx_$m && cp $SP/xbar_ptx/final/.ysu_hw_profile . && cp $X/tests/gemm_f16_512.ysu . && $X/target/release/Y gemm_f16_512.ysu --emit-ptx -o g512.ptx > /dev/null 2>&1)
  cmp -s $L/ptx_$m/g512.ptx $SP/xbar_ptx/final/gemm_f16_512.ptx && same=same || same=differs
  Y_LIB_PATH=$X/target/release/liby.so PYTHONPATH=python timeout 1200 python3 -m unittest discover -s python/y_lang/tests -p test_f16_gemm.py > $L/$m.test.log 2>&1
  rc=$?
  echo "$m: tests exit $rc; 512 PTX $same as the real one; $(grep -E '^(FAILED|OK)' $L/$m.test.log | tail -1); failing: $(grep -E '^(FAIL|ERROR):' $L/$m.test.log | sed -E 's/^(FAIL|ERROR): (test_[a-z_]+).*/\2/' | sort | uniq -c | tr -s ' ' | tr '\n' ';')" >> $L/summary.txt
done
cat $L/ptx_emitter.rs.orig > src/ptx_emitter.rs; touch src/ptx_emitter.rs
nice -n 19 cargo build --release > $L/restore.build.log 2>&1
cmp src/ptx_emitter.rs $L/ptx_emitter.rs.orig && echo "RESTORED, build exit $?" >> $L/summary.txt
