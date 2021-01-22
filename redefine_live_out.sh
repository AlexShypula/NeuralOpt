#!/bin/bash

PATH_TO_DISASSEMBLY_DIR="/home/stoke/docker/11_20_data/"
PATH_TO_IN_STATS_DF="/home/stoke/docker/1_7_redefine_live_out/stats_0_to_350k_1115_collect_1_7_incorrect_to_redefine.csv"
PATH_TO_OUT_STATS_DF="/home/stoke/docker/1_7_redefine_live_out/stats_0_to_350k_1115_collect_1_7_incorrect_redefined.csv"
PATH_TO_SPURIOUS_PROGS="/home/stoke/docker/spurious_progs"
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
  --timeout $TIMEOUT \
  --debug
