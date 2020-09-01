#!/bin/bash
python3 utils/redefine_live_out.py \
  --path_to_disassembly_dir ../ \
  --path_to_in_stats_df ../docker/8_17_costfn/stats_200k.csv \
  --path_to_out_stats_df ../docker/8_17_costfn/stats_redefined.csv \
  --optimized_flag "Og" \
  --debug
