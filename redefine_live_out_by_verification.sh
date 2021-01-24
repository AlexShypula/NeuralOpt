#!/bin/bash

PATH_TO_INPUT_DATAFRAME=""
PATH_TO_OUTPUT_DATAFRAME=""
ALIASING_STRATEGY="basic"
BOUND="2"
VERIFICATION_TIMEOUT="120"
COST_TIMEOUT="300"
N_THREADS="10"
TESTCASES_DIR="testcases"

python3 utils/redefine_live_out_by_verification.py \
  --path_to_input_dataframe $PATH_TO_INPUT_DATAFRAME \
  --path_to_output_dataframe $PATH_TO_OUTPUT_DATAFRAME\
  --aliasing_strategy $ALIASING_STRATEGY \
  --bound $BOUND \
  --verification_timeout $VERIFICATION_TIMEOUT \
  --cost_timeout $COST_TIMEOUT \
  --n_threads $N_THREADS \
  --testcases_dir $TESTCASES_DIR \
#  --debug
