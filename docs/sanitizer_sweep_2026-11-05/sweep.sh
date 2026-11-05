#!/bin/bash
# compute-sanitizer TOOL over every test binary that launches a Y kernel.
#
#   docs/sanitizer_sweep_2026-11-05/sweep.sh memcheck [OUT_DIR]
#
# Runs from the repository root, one binary at a time (--test-threads=1),
# with no print limit, and writes per binary a log and one summary line:
# the sanitizer's own ERROR SUMMARY and a count of each record TYPE. Y's
# runtime checks trap on purpose; a trap and the launch failures it causes
# are records too, so a binary is clean of memory errors when its count of
# `Invalid ...` / `Uninitialized ...` / race / barrier records is zero, and
# the per-type counts must add up to the ERROR SUMMARY (the first record of a
# test lands on the harness's `test name ...` line, so match anywhere).
# The zk_gpu_* binaries run nothing unless built with --features zk.
set -u
tool="${1:?memcheck|racecheck|synccheck|initcheck}"
out="${2:-target/sanitizer_$tool}"
here="$(cd "$(dirname "$0")" && pwd)"
san="$(command -v compute-sanitizer || echo /opt/cuda/bin/compute-sanitizer)"
mkdir -p "$out"
: > "$out/summary.txt"
while read -r t; do
  bin=$(ls -t target/release/deps/"$t"-* 2>/dev/null | grep -v '\.d$' | head -1)
  if [ -z "$bin" ]; then echo "$t NOT BUILT (cargo test --release --test $t --no-run)" >> "$out/summary.txt"; continue; fi
  start=$(date +%s)
  timeout 1800 "$san" --tool "$tool" --print-limit 100000000 --show-backtrace no "$bin" --test-threads=1 > "$out/$t.log" 2>&1
  rc=$?
  sum=$(grep -oE "(ERROR|RACECHECK) SUMMARY: [^|]*" "$out/$t.log" | tail -1)
  kinds=$(grep -oE "========= (Trace/breakpoint trap|Invalid [^ ]+ [a-z]+|Uninitialized [^ ]+ [a-z]+ [a-z]+|Barrier error[^.]*|(Error|Warning): Race reported|Program hit [A-Z_]+|CUDA API Error[^.]*)" "$out/$t.log" \
          | sed 's/========= //' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  res=$(grep -E "^test result" "$out/$t.log" | tail -1 | cut -c13-60)
  echo "$t rc=$rc $(( $(date +%s) - start ))s | $sum | $kinds | $res" >> "$out/summary.txt"
done < "$here/binaries.txt"
echo DONE >> "$out/summary.txt"
