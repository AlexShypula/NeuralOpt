#!/bin/bash
python3 NeuralOpt/utils/stoke_test_costfn.py \
  -path_list demo_successful_paths.txt \
  --unopt-prefix O0 \
  --opt-prefix Og \
  -stats_out demo_costfn/demo_stats.csv \
  -tc_gen_log demo_costfn/demo_tc_gen.log \
  -cost_log demo_costfn/demo_cost.log \
  -benchmark_log demo_costfn/demo_benchmark.log \
  --n-workers 8 \
  -time \
  -stdout_to_csv \
  -write_asbly \
  -benchmark_iters 250 \
  -max_tc 1024 \
  -spm_model_path 4_30_processed/bpe/bpe_1000.model
