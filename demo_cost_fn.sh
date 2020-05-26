#!/bin/bash
python3 utils/stoke_test_costfn.py -path_list demo_successful_paths.txt -unopt_prefix O0 -opt_prefix Og -stats_out demo_costfn/demo_stats.csv -tc_gen_log demo_costfn/demo_tc_gen.log -cost_log demo_costfn/demo_cost.log -benchmark_log demo_costfn/demo_benchmark.log -n_workers 1
