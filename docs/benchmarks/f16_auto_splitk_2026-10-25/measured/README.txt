`Y <shape>.ysu --emit-ptx --autotune` on `_splitk` kernels, each run in a fresh copy of the re-tuned profile:
the compiler's output (`*.log`, with the tuner's tables) and the entries it stored (`entries.txt`).
run1: the first selector (a tie within the noise to the unsplit kernel, then to the fewer splits).
run2: the selector as committed (a tie to the unsplit kernel only), at the two shapes where it differs.
