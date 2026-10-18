#!/bin/bash
# run.sh <tag>: one run of probe_ca2.py with the 957899d compiler snapshot.
P=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad
cd "$P/big/ca2" || exit 9
Y_LIB_PATH="$P/snap957/target/release/liby.so" PYTHONPATH="$P/snap957/python" python3 "$P/big/ca2/probe_ca2.py" \
  "/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y/python/examples" "$P/f16sq" > "$P/big/ca2/ca2_$1.log" 2>&1
echo "ca2 $1 exit $?" >> "$P/big/ca2/summary.txt"
