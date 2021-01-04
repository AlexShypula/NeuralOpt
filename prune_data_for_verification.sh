#!/bin/bash

PATH_TO_DATA_FILES_DIR="/home/stoke/docker/11_20_data/model_data/"
PATH_TO_HASH2METADATA="/home/stoke/docker/11_20_data/model_data/hash2metadata.json"
ASM_PATH_PREFIX="/home/stoke/docker/11_20_data/"
PATH_TO_ERROR_LOG="/home/stoke/11_20_val_data_verify.log"
IN_FILE_PREFIX="test"
OUT_FILE_PREFIX="test_z3_verified"
SRC_FILE_SUFFIX="src"
TGT_FILE_SUFFIX="tgt"

python3 utils/prune_data_for_verified.py \
  --path_to_data_dir $PATH_TO_DATA_FILES_DIR \
  --path_to_hash2metadata $PATH_TO_HASH2METADATA \
  --asm_path_prefix $ASM_PATH_PREFIX \
  --path_to_error_log $PATH_TO_ERROR_LOG \
  --in_file_prefix $IN_FILE_PREFIX \
  --out_file_prefix $OUT_FILE_PREFIX \
  --src_file_suffix $SRC_FILE_SUFFIX \
  --tgt_file_suffix $TGT_FILE_SUFFIX
