#!/bin/bash
P=/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/8ed295d8-e66f-4af6-8439-a5fa065fbc5d/scratchpad/graph_proto
cd /tmp/y_exact_nospace/tools
Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace nice /tmp/y_exact_nospace/venv/bin/python3 -u -I $P/graph_decode_proto.py "$@" 2>&1 | grep -v "^USDT\|Loading weights\|vec512\|0xff\|\^~\|overflow in conv" | tail -40
