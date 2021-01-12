#!/bin/bash

PATH_TO_DEST=""
PATH_TO_DISASSEMBLY_DIR=""
PATH_TO_STATS_CSV=""
PATH_TO_SPM_MODEL=""

PATH_TO_TRAIN_PATHS=""
PATH_TO_DEV_PATHS=""
PATH_TO_TEST_PATHS=""
TC_DIR_NAME="testcases"
REMOVE_FIRST_N_DIRS_IN_PATH=0


#if [ ! -d PATH_TO_DEST ]
#then
#  mkdir PATH_TO_DEST
#fi
#
#if [ ! -d "${PATH_TO_DEST}/disassembly" ]
#then
#  mkdir "${PATH_TO_DEST}/disassembly"
#fi
#
#if [ ! -d "${PATH_TO_DEST}/model_data" ]
#then
#  mkdir "${PATH_TO_DEST}/disassembly"
#fi


python3 utils/make_data.py --path_to_destination_data "${PATH_TO_DEST}/disassembly"\
  --path_to_source_data $PATH_TO_DISASSEMBLY_DIR \
  --path_to_stats_csv $PATH_TO_STATS_CSV \
  --path_to_model_data "${PATH_TO_DEST}/model_data" \
  --path_to_spm_model $PATH_TO_SPM_MODEL \
  --path_to_train_paths $PATH_TO_TRAIN_PATHS \
  --path_to_dev_paths $PATH_TO_DEV_PATHS \
  --path_to_test_paths $PATH_TO_TEST_PATHS \
  --tc_dir_name $TC_DIR_NAME \
  -n_threads 16 \
  --optimize_flag "Og" \
  --remove_first_n_dirs $REMOVE_FIRST_N_DIRS_IN_PATH

# --copy_data_to_dest


### Explanation of parameters

#  --path_to_destination_data -> root folder where you want to save the assembly \
#  --path_to_source_data -> root folder where all the assembly comes from \
#  --path_to_stats_csv -> path to the csv of all training examples after redefine_live_out.py \
#  --path_to_model_data -> path to folder where you will save train.src, train.tgt, etc... \
#  --path_to_spm_model -> sentencepiece model for preprocessing input \
#  --path_to_train_paths -> path to .txt file containing the binary identities of train \
#  --path_to_dev_paths -> path to .txt file containing the binary identities of dev \
#  --oath_to_test_paths -> path to .txt file containing the binary identities of test \
#  -n_threads 16 \
#  --optimize_flag "Og"