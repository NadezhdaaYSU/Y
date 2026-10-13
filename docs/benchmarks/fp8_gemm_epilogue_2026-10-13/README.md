# FP8 GEMM epilogue through shared memory, 2026-10-13

Y's `@tile(N, N, N)` FP8 GEMM from the compiler at `b664885` ("before"; the
kernel of `../fp8_gemm_async_2026-10-12/`) and with this change ("after"),
against the Triton kernel of `python/examples/run_head_to_head_comparison.py`.
All three take float32 A and B, quantize them to e4m3 inside the kernel,
multiply on the e4m3 tensor cores and write float32 C. Every column was
checked equal to `A @ B` bit for bit before it was timed. `bench.py` (the
earlier records' script, unchanged) produced `run1.json` and `run2.json`.

Minimum over 15 interleaved rounds (9 at 8192), microseconds, run 1; run 2.
RTX 4070 Ti SUPER, driver 615.71.09, PyTorch 2.13.0+cu130, Triton 3.7.1:

| N | before | after | Triton | after vs before | after vs Triton |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4.7; 4.7 | 3.7; 3.7 | 11.3; 7.0 | 1.27x faster | 0.33x; 0.53x |
| 512 | 9.4; 9.5 | 7.7; 7.9 | 9.5; 9.3 | 1.21-1.22x | 0.81x; 0.85x |
| 768 | 24.0; 24.2 | 22.6; 22.7 | 24.9; 24.8 | 1.06x | 0.91x |
| 1024 | 34.7; 34.5 | 27.3; 27.4 | 32.5; 32.5 | 1.26-1.27x | 0.84x; 0.85x |
| 1536 | 122.2; 122.5 | 107.2; 107.1 | 132.8; 132.9 | 1.14x | 0.81x |
| 2048 | 213.8; 213.0 | 191.8; 191.5 | 200.1; 198.9 | 1.11x | 0.96x |
| 2560 | 426.5; 427.3 | 398.2; 398.1 | 445.2; 443.6 | 1.07x | 0.89x; 0.90x |
| 3072 | 691.2; 694.3 | 662.5; 665.1 | 720.9; 720.9 | 1.04x | 0.92x |
| 4096 | 1529.3; 1514.8 | 1484.1; 1470.6 | 1568.6; 1576.4 | 1.03x | 0.93x-0.95x |
| 8192 | 12141.6; 12364.8 | 11927.6; 12125.2 | 16157.7; 16482.7 | 1.02x | 0.74x |

No game was running. Chromium and the desktop compositor were the GPU's
other clients: between runs `nvidia-smi` read it 8-11% busy at its 210 MHz
idle clock. It read 31% and 33% busy as each run started (`gpu_busy_before`),
and 13-52% per size, readings taken right after that size's warm-up and
Triton's autotuning. Triton's 256 time differs between the runs (11.3 against 7.0 us)
with the same configuration; a Triton call from Python costs 5.7 us of CPU
time (`../launch_2026-10-12/`), more than its kernel takes at that size.

## What changed

`profile/before_1024.ncu-repz` profiles both kernels at 1024^3 with Triton's
configuration pinned (`profile/prof.py`). Under Nsight Compute's clocks Y took
37.5 us, Triton 35.2. `profile/regions.py` splits each kernel's warp-stall
samples at its K loop:

| | before the loop | loop | after it | global stores |
| :--- | ---: | ---: | ---: | :--- |
| Y before | 6.6% | 66.3% | 27.1% | 32,768 `STG.E` |
| Triton | 2.7% | 89.3% | 8.1% | 8,192 `STG.E.128` |
| Y after (`after_1024.ncu-repz`) | 8.4% | 83.5% | 8.1% | 8,192 `STG.E.128` |

Y wrote each value of C with its own 4-byte store: 64 a thread at 128x128,
each fragment row on its own line. The address arithmetic between the
stores stalled on registers the pending stores still held (long-scoreboard
stalls: 3.37 cycles per issued instruction, Triton's 0.29).

- The cp.async path now writes C through the shared memory its stages are
  done with (`emit_fp8_epilogue_staged`). Each thread stores its scaled
  fragments as column pairs into a tile whose rows are `cta_n + 8` floats
  apart, so a half-warp's pair stores cover the 32 banks once. After a
  barrier each thread copies 16-byte chunks out, consecutive threads taking
  consecutive chunks of a row: a warp writes 512 contiguous bytes at 128
  columns. At 128x128 that is 16 stores a thread, at 64x64 eight.
- The register path (768^3, and small shapes with more than one wave of
  CTAs) stores column pairs with `st.global.v2`: on the padded path
  `n % 4 == 0`, so a pair is in range or out of it whole and 8-byte aligned.

In the after profile Y takes 30.0 us against Triton's 35.4.

## Reproduce

    python3 bench.py emit <checkout-before> before/
    python3 bench.py emit <checkout-after> after/
    PYTHONPATH=<checkout>/python python3 bench.py time before/ after/ results.json
    cd profile && Y_REPO=<checkout> ncu --profile-from-start off --set full \
        --import-source yes -o out python3 prof.py 1024 && python3 regions.py out.ncu-repz
