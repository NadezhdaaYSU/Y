The unsplit kernel on the tile `--autotune` measured for each shape (unsplit_tuned, pinned with
Y_CTA_OVERRIDE) against the profile's tile (unsplit) and the splits, with the main sweep's run_speed.sh:
l2/ with A and B reused, dram/ with DRAM=1. The tuned tiles are the first rows of the unsplit measurement
in ../followup's `--autotune` runs: 64x64x64 4x2 (s3 at K = 4096 and 65536, s4 at 16384) at 512 x 512,
16x64x64 1x2 s4 at 256 x 256 x 16384.
