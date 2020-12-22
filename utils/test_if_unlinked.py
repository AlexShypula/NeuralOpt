from stoke_disassembly import EXECUTABLE_FILE_TAG, _check_executable_fn, COLLAPSE_PATTERN, collapse_path
from make_data import function_path_to_binary_folder
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import json
import pandas as pd
from os.path import join

@dataclass
class ParseOptions:
    path_to_bin_json: str = field(metadata=dict(args=["-path_to_bin_json", "--path_to_bin_json"]))
    path_to_asm_df: str = field(metadata=dict(args=["-path_to_asm_df", "--path_to_asm_df"]))
    path_to_binary_dir: str = field(metadata=dict(args=["-path_to_binary_dir", "--path_to_binary_dir"]))

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    binary_info_dict = json.load(open(args.path_to_bin_json))
    binary_info_dict = {collapse_path(p): v for p, v in binary_info_dict.items()}
    df = pd.read_csv(args.path_to_asm_df)
    df = df[df["unopt_opt_correctness"]=="yes"][df["unopt_unopt_correctness"]=="yes"]
    paths = [function_path_to_binary_folder(p) for p in df["path_to_function"]]
    n_executable = 0
    not_in_dict = 0
    in_dict = 0
    for path in paths:
        path = path if path in binary_info_dict.keys() else path + ".o"
        if path not in binary_info_dict.keys():
            not_in_dict += 1
            continue
        else:
            in_dict+=1
            n_executable +=_check_executable_fn(join(args.path_to_binary_dir, path))
    print(f"number tested: {len(df)}, of those, we found {in_dict} in the dict and {n_executable} executable"
          f"yielding {(n_executable/in_dict)*100:.2f}% as the percent found executable")





