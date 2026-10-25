# shapes.sh: the shape groups, sourced by gen.sh and run_speed.sh.
# A: fewer unsplit CTAs than SMs, above 32 rows (the rule's candidates); D: decode, M <= 32; B: the rest.
X=${X:-/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad/xbar}
SHAPES_A="512x512x4096 512x512x16384 512x512x65536 256x256x16384 64x4096x4096 64x4096x11008 64x4096x14336"
SHAPES_D=""; for M in 1 8 32; do for N in 4096 14336; do for K in 4096 14336; do SHAPES_D="$SHAPES_D ${M}x${N}x${K}"; done; done; done
SHAPES_B="1024x1024x16384 1024x1024x1024 2048x2048x2048 4096x4096x4096 8192x8192x8192"
for sh in $(cat $X/docs/benchmarks/f16_llm_shapes_2026-10-24/sizes.txt); do case " $SHAPES_A " in *" $sh "*) ;; *) SHAPES_B="$SHAPES_B $sh";; esac; done
