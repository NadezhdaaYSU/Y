# Head to head, 2026-10-11

One run of `python/examples/run_head_to_head_comparison.py` on an RTX 4070 Ti
SUPER, started with the GPU 4% busy (the desktop compositor and Discord).
`report.md` is what the script wrote; `results.json` holds every sample, every
check and the toolchain; `stdout.log` is its console output.

The script ran from commit `38b412a` plus `run.diff`, which is why the report
says "dirty tree". The diff changes only the contention rule and the report's
header line, and is committed with these files. The `y_inductor` note in the
report shows "launches 0" because this version read the backend's summary
before the timed calls; the script now reads it after them.
