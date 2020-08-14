#!/bin/bash
python3 utils/stoke_disassemble.py \
  -binary_dir binaries_O0 \
  -disas_dir stoke_disassembly \
  -opt_flag O0 \
  -db_name dire \
  -successful_path_out successful_paths_O0.txt \
  -collection_name alex_repos \
  -config_file ./database-config.json \
  -n_workers 8 \
  -debug False
#
#python3 utils/stoke_preprocess.py \
#  -path_to_bin stoke \
#  -path_list stoke/processed_binaries_multi_proc/succcessful_paths.txt \
#  -train_fldr stoke/shell_res/train \
#  -dev_fldr stoke/shell_res/dev \
# -test_fldr stoke/shell_res/test \
# -model_fldr stoke/shell_res/bpe