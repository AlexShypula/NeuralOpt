#!/bin/bash
python3 NeuralOpt/utils/stoke_collect_v2.py \
  -bin_file_out 8_14_O0.json \
  -db_name dire \
  -field alex_repos \
  -config ./database-config.json \

python3 NeuralOpt/utils/stoke_collect_v2.py \
  -bin_file_out 8_14_Og.json \
  -db_name dire \
  -field alex_repos_g \
  -config ./database-config.json \
#
#python3 utils/stoke_preprocess.py \
#  -path_to_bin stoke \
#  -path_list stoke/processed_binaries_multi_proc/succcessful_paths.txt \
#  -train_fldr stoke/shell_res/train \
#  -dev_fldr stoke/shell_res/dev \
# -test_fldr stoke/shell_res/test \
# -model_fldr stoke/shell_res/bpe
