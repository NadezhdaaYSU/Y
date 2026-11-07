#!/usr/bin/env bash
# ncu on Y's FP32 GEMM: odd N against the nearest N % 4 == 0, two shapes.
set -u
cd "$(dirname "$0")"
M="gpu__time_duration.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
lts__t_sectors_op_write.sum,dram__bytes_write.sum,dram__bytes_read.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active"
ncu --kernel-name regex:^y_ --launch-skip 0 --launch-count 12 --metrics "$M" \
  python3 prof.py 1024x50257x768 1024x50176x768 1024x32001x4096 1024x32000x4096 2>&1
