import subprocess
import re
import pandas as pd
from make_data import function_path_to_optimized_function, function_path_to_testcases, function_path_to_functions_folder
from typing import List
from stoke_test_costfn import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from os.path import join


LIVE_OUT_REGEX = re.compile("(?<=(WARNING: No live out values provided, assuming {))[^}]+")
DEF_IN_REGEX = re.compile("(?<=(WARNING: No def in values provided; assuming {))[^}]+")


@dataclass
class ParseOptions:
    path_to_disassembly_dir: str = field(metadata=dict(args=["-disas_dir", "--path_to_disassembly_dir"]))
    path_to_stats_df: str = field(metadata=dict(args=["-in_stats_df", "--path_to_in_stats_df"]))
    path_to_out_stats_df: str = field(metadata=dict(args=["-out_stats_df", "--path_to_out_stats_df"]))
    optimized_flag: str = field(metadata=dict(args=["-optimized_flag", "--optimized_flag"]), default = "Og")
    debug: bool = field(metadata=dict(args=["-d", "--debug"]), default=False)


def register_list_from_regex(stoke_diff_string: str, pattern: re.Pattern):
    match = pattern.search(stoke_diff_string)
    string = match.group()
    register_list = string.strip().split(" ")
    return register_list


def register_list_to_register_string(register_list: List[str]):
    s = " ".join(register_list)
    return " ".join(["{", s, "}"])


def stoke_diff_get_live_out(def_in_register_list: List[str], live_out_register_list: List[str],
               target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, live_dangerously: bool = False):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    try:
        diff = subprocess.run(
            ["stoke", "debug", "diff", "--target", target_f, "--rewrite", rewrite_f, "--testcases",
             testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str,
             "--live_out", register_list_to_register_string(live_out_register_list),
             "--def_in", register_list_to_register_string(def_in_register_list)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25
        )
        new_live_out_list = []

        if diff.returncode == 0:
            for register in live_out_register_list:
                if not re.search(register, diff.stdout):
                    new_live_out_list.append(register)

        return diff.returncode, diff.stdout, new_live_out_list

    except subprocess.TimeoutExpired as err:
        return -1, err, []

def redefine_live_out_df(path_to_disassembly_dir: str, df: pd.DataFrame, optimized_flag = "Og", debug = False):
    new_rows = []
    for i, row in df.iterrows():
        def_in_register_list = register_list_from_regex(opt_cost_string, DEF_IN_REGEX)
        live_out_register_list = register_list_from_regex(opt_cost_string, LIVE_OUT_REGEX)
        if row["unopt_unopt_correctness"] == "yes":
            if row["opt_unopt_correctness"] == "yes":
                row["heap_out"] = True  # pre-set to true as default
                row["def_in"] = register_list_to_register_string(def_in_register_list)
                row["live_out"] = register_list_to_register_string(live_out_register_list)
                new_rows.append(row.to_dict())
            else:
                path_to_function = join(path_to_disassembly_dir,  row["path_to_function"])
                opt_cost_string = row["opt_cost_str"]

                functions_dir = function_path_to_functions_folder(path=path_to_function)
                path_to_optimized_function = function_path_to_optimized_function(path = path_to_function,
                                                                                 optimized_flag = optimized_flag)
                path_to_testcases = function_path_to_testcases(path = path_to_function, tc_folder = "testcases")

                diff_rc, diff_stdout, live_out_register_list = stoke_diff_get_live_out(def_in_register_list=def_in_register_list,
                                                                                       live_out_register_list=live_out_register_list,
                                                                                       target_f=path_to_function,
                                                                                       rewrite_f=path_to_optimized_function,
                                                                                       testcases_f=path_to_testcases,
                                                                                       fun_dir = functions_dir,
                                                                                       live_dangerously=True)


                if diff_rc == 0:
                    cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                                         rewrite_f=path_to_optimized_function,
                                                                         testcases_f=path_to_testcases,
                                                                         fun_dir=functions_dir,
                                                                         def_in_register_list=def_in_register_list,
                                                                         live_out_register_list=live_out_register_list,
                                                                         heap_out=True)
                    if correct_str == "yes":
                        row["heap_out"] = True
                        row["opt_unopt_cost"] = float(cost)
                        row["opt_unopt_correctness"] = correct_str
                        row["def_in"] = register_list_to_register_string(def_in_register_list)
                        row["live_out"] = register_list_to_register_string(live_out_register_list)
                        new_rows.append(row.to_dict())
                    elif correct_str == "no":
                        row["heap_out"] = False  # pre-set to true as default
                        # try to then suppress the heap
                        cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                                              rewrite_f=path_to_optimized_function,
                                                                              testcases_f=path_to_testcases,
                                                                              fun_dir=functions_dir,
                                                                              def_in_register_list=def_in_register_list,
                                                                              live_out_register_list=live_out_register_list,
                                                                              heap_out=False)

                        if correct_str == "yes":
                            row["opt_unopt_cost"] = float(cost)
                            row["opt_unopt_correctness"] = correct_str
                            row["def_in"] = register_list_to_register_string(def_in_register_list)
                            row["live_out"] = register_list_to_register_string(live_out_register_list)
                            new_rows.append(row.to_dict())

                        else:
                            if debug:
                                breakpoint()
                            continue
                    else:
                        if debug:
                            breakpoint()
                        raise ValueError(f"correct str needs to be 'yes' or 'no' it was {correct_str}")
                else:
                    if debug:
                        breakpoint()
                    continue
    df = pd.DataFrame(new_rows)
    return df


def test_costfn(target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str,
                def_in_register_list: List[str], live_out_register_list: List[str], heap_out: bool,
                live_dangerously: bool = True ):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    heap_out_str = "--heap_out" if heap_out else ""
    try:
        cost_test = subprocess.run(
            ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
             testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
             '{ 0 1 ... 31 }', '--cost', '100*correctness+measured+latency', heap_out_str
             '--def_in', register_list_to_register_string(def_in_register_list),
             '--live_out', register_list_to_register_string(live_out_register_list)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25)
        if cost_test.returncode == 0:
            cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
            correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()
        else:
            cost = -10701
            correct = "failed"
        return cost_test.returncode, cost_test.stdout, cost, correct

    except subprocess.TimeoutExpired as err:
        return -11785, err, -11785, "timeout"


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    df_in = pd.read_csv(args.path_to_stats_df)
    df_out = redefine_live_out_df(path_to_disassembly_dir=args.path_to_disassembly_dir,
                                  df = df_in,
                                  optimized_flag=args.optimized_flag,
                                  debug=args.debug)
    df_out.to_csv(args.path_to_out_stats_df)

