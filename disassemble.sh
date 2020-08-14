#/bin/bash
python3 NeuralOpt/utils/stoke_disassemble.sh \
  -bin_file_in  docker/8_14_O0.json\
  -binary_dir docker/binaries_O0 \
  -disas_dir docker/disassembly \
  -opt_flag O0 \
  -successful_paths_out_file 8_14_successful_paths \
  -n_workers 4 \
  -debug