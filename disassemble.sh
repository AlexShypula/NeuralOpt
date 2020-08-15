#/bin/bash
python3 NeuralOpt/utils/stoke_disassemble.py \
  -bin_file_in  docker/8_14_O0.json\
  -binary_dir docker/binaries_O0 \
  -disas_dir docker/disassembly \
  -opt_flag O0 \
  -successful_path_out 8_14_successful_paths_O0.txt \
  -n_workers 8

python3 NeuralOpt/utils/stoke_disassemble.py \
  -bin_file_in  docker/8_14_Og.json\
  -binary_dir docker/binaries_Og \
  -disas_dir docker/disassembly \
  -opt_flag Og \
  -successful_path_out 8_14_successful_paths_Og.txt \
  -n_workers 8 
