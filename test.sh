#!/bin/bash
python3 model/prediction.py \
    -config configs/stoke_7_20_Og.yaml \
    --path_to_checkpoint ../joeynmt/models_stoke/7_20_asm_0/best.ckpt \
    --nbest_beams 5 \
    -beam_size 5 \
    -output_path ../demo_testing/ \
