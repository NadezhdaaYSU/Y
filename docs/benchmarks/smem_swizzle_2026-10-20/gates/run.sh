#!/bin/bash
# run.sh <tree> <label> <expected ptx_emitter.rs>: the validator's gates on one
# tree, at nice 19, holding the GPU lock. The tree must hold the expected
# source (not a mutant), freshly built.
SP=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
L=$SP/tval_s
exec 9>"$SP/gpu.lock"; flock 9
cmp "$1/src/ptx_emitter.rs" "$3" || { echo "$2 SOURCE IS NOT THE EXPECTED ONE" >> "$L/summary.txt"; exit 8; }
(cd "$1" && nice -n 19 cargo build --release > "$L/build_$2.log" 2>&1) || { echo "$2 BUILD FAILED" >> "$L/summary.txt"; exit 7; }
cd "$1/tools/ptxas_tval" || exit 9
export PATH="$1/target/release:$PATH"
for g in "./build_corpus.sh" "./regress.sh" "python3 fpgate.py" "python3 docgate.py" "python3 gap.py" "python3 unroll.py"; do
  n=$(echo "$g" | sed 's/[^a-z_]//g; s/python3//; s/sh$//')
  s=$(date +%s); timeout 3600 nice -n 19 $g > "$L/${n}_$2.log" 2>&1; echo "$2 $g exit $? $(( $(date +%s)-s ))s" >> "$L/summary.txt"
done
echo "$2 DONE" >> "$L/summary.txt"
