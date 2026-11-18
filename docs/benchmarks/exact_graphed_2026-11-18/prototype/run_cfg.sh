#!/bin/bash
# usage: run_cfg.sh <every|default> <ragged 0|1>
P=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8ed295d8-e66f-4af6-8439-a5fa065fbc5d/scratchpad/graph_proto
cd /tmp/y_exact_nospace/tools
if [ "$1" = every ]; then export Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace; fi
RAGGED=$2 REPS=3 nice /tmp/y_exact_nospace/venv/bin/python3 -u -I $P/graph_both.py 2>&1 | grep -E "graphed ==|tok/s|x$|Error|error|Traceback" | head -20
