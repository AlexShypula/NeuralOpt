#!/bin/bash
python3 model/prediction.py \
    -config configs/stoke_7_20_Og.yaml \
    --path_to_checkpoint ../joeynmt/models_stoke/7_20_asm_0/best.ckpt \
    --nbest_beams 5 \
    -beam_size 5 \
    -output_path ../9_2_beams/beam_5_pretrain \

python3 model/prediction.py \
    -config configs/stoke_7_20_Og.yaml \
    --path_to_checkpoint ../joeynmt/models_stoke/7_20_asm_0/best.ckpt \
    --nbest_beams 10 \
    -beam_size 10 \
    -output_path ../9_2_beams/beam_10_pretrain \

python3 model/prediction.py \
    -config configs/stoke_7_20_Og.yaml \
    --path_to_checkpoint ../7_20_asm_0_Og_8_29/best.ckpt \
    --nbest_beams 5 \
    -beam_size 5 \
    -output_path  ../9_2_beams/beam_5_rl  \

python3 model/prediction.py \
    -config configs/stoke_7_20_Og.yaml \
    --path_to_checkpoint ../7_20_asm_0_Og_8_29/best.ckpt \
    --nbest_beams 10 \
    -beam_size 10 \
    -output_path  ../9_2_beams/beam_10_rl  \

