#!/bin/bash

python3 model/predict.py --path_to_config insert_path_to_config_here \
  --path_to_checkpoint insert_ckpt_path_here \
  --nbest_beams 10 \
  --beam_size 10 \
  --api_ip_address 127.0.0.1 \
  --api_port_no 6001 \
  --datasets_to_test dev:train \
  --experiment_name insert_experiment_name_here
  ## --output_path ## in case you want to save it elsewhere besides model folder