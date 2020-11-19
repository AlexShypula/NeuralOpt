#!/bin/bash
EXPERIMENT_NAME="EXPERIMENT_NAME_HERE"
DATASETS_TO_TEST="dev:train"

for CHECKPOINT in path_to_ckpt1 path_to_ckpt2 path_to_ckpt3
do
  python3 model/predict.py --path_to_config insert_path_to_config_here \
  --path_to_checkpoint $CHECKPOINT \
  --nbest_beams 10 \
  --beam_size 10 \
  --api_ip_address 127.0.0.1 \
  --api_port_no 6001 \
  --datasets_to_test $DATASETS_TO_TEST \
  --experiment_name $EXPERIMENT_NAME
done

#### an additional argument we can specify in the args is here:
## --output_path ## in case you want to save it elsewhere besides model folder