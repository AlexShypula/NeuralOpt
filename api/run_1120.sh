#!/bin/bash
python3 NeuralOpt/api/api.py -path_to_volume docker -path_to_tmp tmp_train_O0 -path_to_data 11_20_data -n_workers 11 -max_cost 120000 -port 6001
