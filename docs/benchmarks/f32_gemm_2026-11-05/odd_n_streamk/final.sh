#!/usr/bin/env bash
# The record's final runs. usage: final.sh Y_OLD Y_NEW Y_SK_FIRST
# Y_OLD: 83da64d's build; Y_SK_FIRST: stream-K at N odd with the old B clamp (the first build).
# Idle GPU. Writes sweep_run{1,2}.txt, streamk_run{1,2}.txt, cta_times.txt here.
set -u
OLD="$(realpath "$1")"; NEW="$(realpath "$2")"; SKF="$(realpath "$3")"
H="$(realpath "$(dirname "$0")")"
PY="$H/../../../../venv/bin/python"
PLAIN="1024x1025x1024 1023x1025x1027 1025x1025x1025 1024x1021x1024 1024x1028x1024 1024x897x1024 1000x1001x1000 768x1025x768 2048x2049x1024 2047x2049x2051 4095x4097x512 1024x50257x768 1024x30522x768 1024x1024x1024"
SK="1023x1025x1027 1024x1025x1024 1025x1025x1025 1100x1101x1100 640x1025x2048 1024x1153x1024 512x1025x4096 1000x1153x2000 960x1281x960 768x1025x768 2048x1025x1024 1024x1028x1024 768x768x768"
for i in 1 2; do
  "$PY" "$H/sweep.py" "$OLD=old" "$NEW=new" -- $PLAIN 2>&1 | grep -v -i warn > "$H/sweep_run$i.txt"
done
W=$(mktemp -d); cp "$H/../../../../.ysu_hw_profile" "$W/"; cd "$W"   # streamk_bench compiles in its cwd; without the profile the compiler re-probes and plans for what it finds
for i in 1 2; do
  YBIN="$NEW" "$PY" "$H/../streamk_bench.py" $SK 2>&1 | grep -v -i warn > "$H/streamk_run$i.txt"
done
{ for b in "$SKF" "$NEW"; do for s in 1024x1025x1024 1024x1028x1024; do echo "[$( [ "$b" = "$SKF" ] && echo "first stream-K build, B past N clamped to the last chunk" || echo "final build" )]"; "$PY" "$H/cta_times.py" "$b" $s 2>&1 | grep -v -i warn; done; done; } > "$H/cta_times.txt"
