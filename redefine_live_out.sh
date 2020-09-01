#!/bin/bash
python3 utils/redefine_live_out.py \
  --path_to_disassembly_dir \
  --path_to_in_stats_df \
  --path_to_out_stats_df \
  --optimized_flag "Og" \
  --debug
