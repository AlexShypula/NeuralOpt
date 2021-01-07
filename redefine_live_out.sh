#!/bin/bash

PATH_TO_DISASSEMBLY_DIR=""
PATH_TO_IN_STATS_DF=""
PATH_TO_OUT_STATS_DF=""
PATH_TO_SPURIOUS_PROGS=""
SPURIOUS_PROG_LIST="eax_zero.s:rax_zero.s:retq.s"
OPTIMIZED_FLAG="Og"
N_WORKERS="8"
TIMEOUT="120"

python3 utils/redefine_live_out.py \
  --path_to_disassembly_dir $PATH_TO_DISASSEMBLY_DIR \
  --path_to_in_stats_df $PATH_TO_IN_STATS_DF \
  --path_to_out_stats_df $PATH_TO_OUT_STATS_DF \
  --path_to_spurious_dir $PATH_TO_SPURIOUS_PROGS \
  --spurious_prog_list $SPURIOUS_PROG_LIST \
  --optimized_flag $OPTIMIZED_FLAG \
  --n_workers $N_WORKERS \
  --timeout $TIMEOUT
# --debug
