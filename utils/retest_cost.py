import subprocess
import re
import pandas as pd
from copy import copy, deepcopy
from make_data import function_path_to_optimized_function, function_path_to_testcases, function_path_to_functions_folder
from typing import List, Dict
from stoke_test_costfn import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from os.path import join
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from registers import DEF_IN_REGISTER_LIST, LIVE_OUT_REGISTER_LIST, REGISTER_TO_STDOUT_REGISTER, \
    GP_REGISTERS_SET, LIVE_OUT_FLAGS_SET, gp_reg_64_to_32, gp_reg_64_to_16, gp_reg_64_to_8
from make_data import function_path_to_functions_folder, function_path_to_testcases
from redefine_live_out import test_costfn


@dataclass
class ParseOptions:
    path_to_disassembly_dir: str = field(metadata=dict(args=["-disas_dir", "--path_to_disassembly_dir"]))
    path_to_stats_df: str = field(metadata=dict(args=["-in_stats_df", "--path_to_in_stats_df"]))
    path_to_out_stats_df: str = field(metadata=dict(args=["-out_stats_df", "--path_to_out_stats_df"]))
    n_workers: int = field(metadata=dict(args=["-n_threads"]), default=8
    debug: bool = field(metadata=dict(args=["-d", "--debug"]), default=False)

REGISTER_LIST_REGEX = re.compile("(?<=({))[^}]+")


def register_list_from_string(register_list_string: str, pattern):
    match = pattern.search(register_list_string)
    registers = match.strip().split()
    return registers


def test_individual_row(row: Dict, path_to_disassembly_dir: str):
    path = row["path_to_function"]
    full_path_to_function = join(path_to_disassembly_dir, path)
    path_to_functions_dir = function_path_to_functions_folder(full_path_to_function)
    path_to_testcases = function_path_to_testcases(full_path_to_function)
    def_in_register_list = register_list_from_string(row["def_in"])
    live_out_register_list = register_list_from_string(row["live_out"])
    stack_out = row["stack_out"]
    heap_out = row["heap_out"]


    cost_test_rc, cost_test_stdout, cost, correct = test_costfn(target_f=full_path_to_function,
                                                                rewrite_f=full_path_to_function,
                                                                testcases_f=path_to_testcases,
                                                                fun_dir=path_to_functions_dir,
                                                                def_in_refister_list=def_in_register_list,
                                                                live_out_register_list=live_out_register_list,
                                                                stack_out=stack_out,
                                                                heap_out=heap_out,
                                                                live_dangerously=True)
    assert correct == "yes"
    row["unopt_unopt_cost"] = cost
    return row

# def test_individual_row_wrapper(args):
#     return test_individual_row(**args)

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()

    stats_dataframe = pd.read_csv(args.path_to_stats_df)
    stats_dataframe = stats_dataframe[stats_dataframe["opt_unopt_correctness"] == "yes"]\
        [stats_dataframe["unopt_unopt_correctness"] == "yes"]
    list_of_row_dicts = stats_dataframe.reindex().to_dict('records')
    jobs = [(row, args.path_to_disassembly_dir) for row in list_of_row_dicts]
    results = []
    if not args.debug:
        for new_row in ThreadPool(args.n_threads).imap(test_individual_row, jobs, chunksize=88):
            results.append(new_row)
    else:
        for new_row in map(args.n_threads).imap(test_individual_row, jobs, chunksize=88):
            results.append(new_row)
    out_df = pd.DataFrame(results)
    out_df["unopt_and_opt_equal"] = out_df["unopt_assembly"] == out_df["opt_assembly"]
    out_df.to_csv(args.path_to_out_stats_df)
