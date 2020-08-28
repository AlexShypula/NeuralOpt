#!/bin/bash
python3 NeuralOpt/api/api.py -path_to_volume docker/ -path_to_tmp tmp_train_O0 -path_to_data "" -n_workers 8 -max_cost 9999 -port 6000
