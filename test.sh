#!/bin/bash
python3 model/prediction.py \
    -config \
    --path_to_checkpoint \
    --nbest_beams \
    -beam_size \
    -beam_alpha \
    -output_path \