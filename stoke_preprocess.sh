#!/bin/bash
python3 utils/stoke_preprocess.py -path_to_bin stoke -path_list stoke/processed_binaries_multi_proc/succcessful_paths.txt -train_fldr stoke/shell_res/train -dev_fldr stoke/shell_res/dev -test_fldr stoke/shell_res/test -model_fldr stoke/shell_res/bpe
