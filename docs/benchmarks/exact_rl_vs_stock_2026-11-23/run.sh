#!/usr/bin/env bash
# Exact RL against stock bf16 PyTorch and vLLM on the same work, on this
# machine's GPU: exact (10 steps, graphed sampler, compiled products), the
# stock arms on its dump, vLLM on its prompts and lengths, exact again.
# Paths go through a link without spaces: torch.compile cannot link from a
# path with one.
set -euo pipefail
REPO="/home/yumin/NVME files/YSU-engine-main/YSU-engine-main/src/Y_lang/Y"
LINK=/tmp/y_exact_nospace
[ -e "$LINK" ] || ln -s "$REPO" "$LINK"
D="$LINK/docs/benchmarks/exact_rl_vs_stock_2026-11-23"
OUT="$D/rtx4070tis"
DATA="$LINK/docs/benchmark_data/exact_rl_vs_stock"
mkdir -p "$OUT" "$DATA"
PY="$LINK/venv/bin/python3"
VPY="$HOME/vllm_env/bin/python"

if pgrep -x sober >/dev/null || pgrep -x steam >/dev/null || pgrep -x roblox >/dev/null; then
    echo "a game is running: not using the GPU" >&2
    exit 1
fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$OUT/gpu_before.txt"

cd "$LINK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXACT=(tools/exact_lora_grpo.py --device cuda --compile --graphed --mode steps --new 128 --early-stop
       --sample-prompts 32 --chunk-tokens 1024 --steps 10 --eval 0 --eval-every 0 --log 1)

nice "$PY" "${EXACT[@]}" --record "$OUT/exact_a.json" --dump "$DATA/exact_a.pt" 2>&1 | tee "$OUT/exact_a.log"
nice "$PY" tools/exact_rl_stock_speed.py "$DATA/exact_a.pt" --rounds 2 --record "$OUT/stock.json" \
    --vllm-input "$DATA/vllm_in.json" 2>&1 | tee "$OUT/stock.log"
nice "$VPY" "$D/vllm_sample.py" "$DATA/vllm_in.json" --record "$OUT/vllm.json" 2>&1 | tee "$OUT/vllm.log"
nice "$PY" "${EXACT[@]}" --record "$OUT/exact_b.json" --dump "$DATA/exact_b.pt" 2>&1 | tee "$OUT/exact_b.log"
"$PY" tools/exact_lora_grpo.py --compare "$OUT/exact_a.json" "$OUT/exact_b.json" | tee "$OUT/exact_ab_compare.txt"
"$PY" "$D/summarize.py" "$DATA/exact_a.pt" "$DATA/exact_b.pt" "$OUT/stock.json" "$OUT/vllm.json" | tee "$OUT/summary.txt"
