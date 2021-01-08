import subprocess
import re
import pandas as pd
import os
import numpy as np
from copy import copy, deepcopy
from make_data import function_path_to_optimized_function, function_path_to_testcases, function_path_to_functions_folder,\
    remove_first_n_dirs
from typing import List
from stoke_test_costfn import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from os.path import join, splitext, basename
from tqdm import tqdm
from multiprocessing import Pool
from registers import DEF_IN_REGISTER_LIST, LIVE_OUT_REGISTER_LIST, REGISTER_TO_STDOUT_REGISTER, \
    GP_REGISTERS_SET, LIVE_OUT_FLAGS_SET, gp_reg_64_to_32, gp_reg_64_to_16, gp_reg_64_to_8, AMD64_ABI_REGISTERS_W_FP, \
    AMD64_ABI_REGISTERS



@dataclass
class ParseOptions:
    path_to_disassembly_dir: str = field(metadata=dict(args=["-disas_dir", "--path_to_disassembly_dir"]))
    path_to_stats_df: str = field(metadata=dict(args=["-in_stats_df", "--path_to_in_stats_df"]))
    path_to_out_stats_df: str = field(metadata=dict(args=["-out_stats_df", "--path_to_out_stats_df"]))
    path_to_spurious_dir: str = field(metadata=dict(args=["-spurious_dir", "--path_to_spurious_dir"]), default=None)
    spurious_progs: str = field(metadata=dict(args=["-spurious_prog_list", "--spurious_prog_list"]), default=None)
    optimized_flag: str = field(metadata=dict(args=["-optimized_flag", "--optimized_flag"]), default = "Og")
    n_workers: int = field(metadata=dict(args=["-n_workers", "--n_workers"]), default = 1)
    debug: bool = field(metadata=dict(args=["-d", "--debug"]), default=False)
    timeout: int = field(metadata=dict(args=["-timeout", "--timeout"]), default=600)
    make_new_testcases: bool = field(metadata=dict(args=["-make_new_testcases", "--make_new_testcases"]), default=Falqse)
    new_tc_dir: str = field(metadata=dict(args=["-new_tc_dir", "--new_tc_dir"]), default=None)
    bound: int = field(metadata=dict(args=["-tc_bound", "--tc_bound"]), default=8)
    max_tcs: int = field(metadata=dict(args=["-max_tcs", "--max_tcs"]), default=256)


ANSI_REGEX = re.compile(r'\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))')
LIVE_OUT_REGEX = re.compile("|".join(["({})".format(r) for r in LIVE_OUT_FLAGS_SET]))


def clean_ansi_color_codes(string_to_clean: str):
    return ANSI_REGEX.sub("", string_to_clean)


def test_if_lower_order_register(match_list: List[str], gp_register_name_64: str):
    assert len(match_list) == 2
    tgt_byte_list = match_list[0].strip().split()
    rewrite_byte_list = match_list[1].strip().split()
    # test if the lower 32 bits match (last 4 bytes)
    if " ".join(tgt_byte_list[-4:]) == " ".join(rewrite_byte_list[-4:]):
        return gp_reg_64_to_32[gp_register_name_64]
    # test if the lower 16 bits match (last 2 bytes)
    elif " ".join(tgt_byte_list[-2:]) == " ".join(rewrite_byte_list[-4:]):
        return gp_reg_64_to_16[gp_register_name_64]
    # test if the lower 8 bits match (last 1 byte)
    elif tgt_byte_list[-1] == rewrite_byte_list[-1]:
        return gp_reg_64_to_8[gp_register_name_64]
    # the entire register is different, so remove from live_out
    else:
        return None


def register_list_from_regex(stoke_diff_string: str, pattern: re.Pattern):
    match = pattern.search(stoke_diff_string)
    string = match.group()
    register_list = string.strip().split(" ")
    return register_list


def register_list_to_register_string(register_list: List[str]):
    s = " ".join(register_list)
    return " ".join(["{", s, "}"])


def stoke_diff_get_live_out(def_in_register_list: List[str], live_out_register_list: List[str],
               target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, live_dangerously: bool = False,
                            debug: bool = False):
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
            if debug:
                print("def_in_register_list: " + " " .join(def_in_register_list))
                print("orig live_out_register_list: " + " ".join(def_in_register_list))
                print("diff std out " + diff.stdout)
                print("new_live_out_list: " + " ".join(new_live_out_list))

        return diff.returncode, diff.stdout, new_live_out_list

    except subprocess.TimeoutExpired as err:
        return -1, err, []

def stoke_diff_get_live_out_v2(def_in_register_list: List[str], live_out_register_list: List[str],
               target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, live_dangerously: bool = False,
                            debug: bool = False, timeout: int = 600):

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
            timeout=timeout
        )
        new_live_out_register_list = []
        
        if re.search("returned abnormally with signal", diff.stdout): 
            diff.returncode = 11

        if diff.returncode == 0:
            diff_stdout_clean = clean_ansi_color_codes(diff.stdout)
            for register in live_out_register_list:
                register_stdout_code = REGISTER_TO_STDOUT_REGISTER[register]
                findall_string = "(?<=(?:{}))[^\n]+".format(register_stdout_code)
                findall_result = re.findall(findall_string, diff_stdout_clean)
                # if there is no regular expressions match (i.e. this register is equal in unopt and opt versions)
                # findall will return an empty list
                if findall_result == []:
                    new_live_out_register_list.append(register)
                # only if it is a general purpose register will we try to search for a lower register to test out
                elif register in GP_REGISTERS_SET:
                    assert len(findall_result) == 2, "findall result is not length 2, result is {} and diff is {}"\
                        .format(findall_result, diff_stdout_clean)
                    lower_order_register_result = test_if_lower_order_register(match_list=findall_result,
                                                                               gp_register_name_64=register)
                    # the test returns a register name if a lower-order register is found, otherwise None
                    if lower_order_register_result != None:
                        new_live_out_register_list.append(lower_order_register_result)
            if debug:
                print("def_in_register_list: " + " " .join(def_in_register_list))
                print("orig live_out_register_list: " + " ".join(def_in_register_list))
                print("diff std out cleaned" + diff_stdout_clean)
                print("new_live_out_list: " + " ".join(new_live_out_register_list))

            # if any flags are in the diff, remove all flags (i.e. don't test FLAGS register)
            if LIVE_OUT_REGEX.search(diff_stdout_clean): 
                new_live_out_register_list = [r for r in new_live_out_register_list if r not in LIVE_OUT_FLAGS_SET]
        # diff did not return with 0 returncode
        else:
            pass

    except subprocess.TimeoutExpired as err:
        return -1, err, []

    return diff.returncode, diff.stdout, new_live_out_register_list

def redefine_live_out_df_wrapper(args):
    return redefine_live_out_df(**args)

def _redefine_live_out_indiv(row: pd.Series, live_out_register_list: List[str], def_in_register_list: List[str],
                        path_to_spurious_dir: str, spurious_program_list: List[str],
                        path_to_disassembly_dir: str, optimized_flag: str, debug: bool, timeout: int,
                        tc_directory: str = "testcases"):

    path_to_function = remove_first_n_dirs(row["path_to_function"], 1)
    path_to_function = join(path_to_disassembly_dir, path_to_function)

    functions_dir = function_path_to_functions_folder(path=path_to_function)
    path_to_optimized_function = function_path_to_optimized_function(path=path_to_function,
                                                                     optimized_flag=optimized_flag)

    path_to_testcases = function_path_to_testcases(path=path_to_function, tc_folder=tc_directory)
    # iteratively re-define live-out
    diff_rc, diff_stdout, live_out_register_list = stoke_diff_get_live_out_v2(def_in_register_list=def_in_register_list,
                                                                              live_out_register_list=live_out_register_list,
                                                                              target_f=path_to_function,
                                                                              rewrite_f=path_to_optimized_function,
                                                                              testcases_f=path_to_testcases,
                                                                              fun_dir=functions_dir,
                                                                              live_dangerously=True,
                                                                              debug=debug)

    if isinstance(diff_stdout, str) and re.search("returned abnormally with signal", diff_stdout):
        diff_rc = 11
    # always do stack out as false
    if diff_rc == 0:
        cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                              rewrite_f=path_to_optimized_function,
                                                              testcases_f=path_to_testcases,
                                                              fun_dir=functions_dir,
                                                              def_in_register_list=def_in_register_list,
                                                              live_out_register_list=live_out_register_list,
                                                              heap_out=True,
                                                              stack_out=False,
                                                              timeout=timeout)
        if correct_str == "yes":
            if path_to_spurious_dir and spurious_program_list:
                is_spurious = is_spurious_program(path_to_spurious_dir=path_to_spurious_dir,
                                                  spurious_program_list=spurious_program_list,
                                                  path_to_function=path_to_function,
                                                  path_to_testcases=path_to_testcases,
                                                  functions_dir=functions_dir,
                                                  def_in_register_list=def_in_register_list,
                                                  live_out_register_list=live_out_register_list,
                                                  heap_out=True, stack_out=False, timeout=timeout)
                correct_str = "spurious" if is_spurious else correct_str
            row["heap_out"] = True
            row["stack_out"] = False
            row["opt_unopt_cost"] = float(cost)
            row["opt_unopt_correctness"] = correct_str
            row["def_in"] = register_list_to_register_string(def_in_register_list)
            row["live_out"] = register_list_to_register_string(live_out_register_list)
            row["opt_cost_str"] = cost_stdout
            return row

        # try to then suppress the heap
        elif correct_str == "no":
            cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                                  rewrite_f=path_to_optimized_function,
                                                                  testcases_f=path_to_testcases,
                                                                  fun_dir=functions_dir,
                                                                  def_in_register_list=def_in_register_list,
                                                                  live_out_register_list=live_out_register_list,
                                                                  heap_out=False,
                                                                  stack_out=False,
                                                                  timeout=timeout)

            if correct_str == "yes":
                if path_to_spurious_dir and spurious_program_list:
                    is_spurious = is_spurious_program(path_to_spurious_dir=path_to_spurious_dir,
                                                      spurious_program_list=spurious_program_list,
                                                      path_to_function=path_to_function,
                                                      path_to_testcases=path_to_testcases,
                                                      functions_dir=functions_dir,
                                                      def_in_register_list=def_in_register_list,
                                                      live_out_register_list=live_out_register_list,
                                                      heap_out=False, stack_out=False, timeout=timeout)
                    correct_str = "spurious" if is_spurious else correct_str
                row["heap_out"] = False
                row["stack_out"] = False
                row["opt_unopt_cost"] = float(cost)
                row["opt_unopt_correctness"] = correct_str
                row["def_in"] = register_list_to_register_string(def_in_register_list)
                row["live_out"] = register_list_to_register_string(live_out_register_list)
                row["opt_cost_str"] = cost_stdout
                return row

            # if debug, then print out what the remaining diff is
            else:
                if debug:
                    diff = subprocess.run(
                        ["stoke", "debug", "diff", "--target", path_to_function, "--rewrite",
                         path_to_optimized_function, "--testcases",
                         path_to_testcases, '--functions', functions_dir, "--prune", "--live_dangerously",
                         "--live_out",
                         register_list_to_register_string(live_out_register_list),
                         "--def_in", register_list_to_register_string(live_out_register_list)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout
                    )
                    print(f"even after redefine live out, it was incorrect, live out is "
                          f"{register_list_to_register_string(live_out_register_list)}")
                    print(f"diff stdout is {diff.stdout}")

            row["opt_unopt_cost"] = float(cost)
            row["opt_unopt_correctness"] = correct_str
            row["def_in"] = register_list_to_register_string(def_in_register_list)
            row["live_out"] = register_list_to_register_string(live_out_register_list)
            row["diff_str"] = diff_stdout
            row["opt_cost_str"] = cost_stdout
            return row

        # if correct_str from test_costfn is neither "yes" nor "no"
        else:
            if debug:
                pass

            row["opt_unopt_correctness"] = correct_str
            return row

    # if diff process did not return 0
    else:
        if debug:
            pass
            # breakpoint()
        row["opt_cost_str"] = diff_stdout
        row["opt_unopt_correctness"] = "error occurred"
        row["def_in"] = register_list_to_register_string(def_in_register_list)
        row["live_out"] = register_list_to_register_string(live_out_register_list)
        row["diff_str"] = diff_stdout
        return row


def redefine_live_out_df(path_to_disassembly_dir: str, df: pd.DataFrame, path_to_spurious_dir, spurious_program_list,
                         optimized_flag = "Og", position: int = 0, debug: bool = False, timeout: int = 600,
                         make_new_testcases: bool = False, new_tc_dir: str = None, bound: int = None,
                         max_tcs: int = None):
    new_rows = []
    pbar = tqdm(total = len(df), position = position)
    for i, row in df.iterrows():
        if row["unopt_unopt_correctness"] == "yes":
            if make_new_testcases:
                assert new_tc_dir, "need to specify testcase dir"
                assert new_tc_dir != "testcases" , "need to specify new testcase dir name that is not 'testcases'"
                row, tc_gen_proc_rc = _tc_gen_symbolic_and_test_cost(row, path_to_disassembly_dir, new_tc_dir,
                                                                     bound, max_tcs, timeout)
                live_out_register_list = register_list_from_string(row["live_out"])
                tc_dir = new_tc_dir
            else:
                live_out_register_list = copy(LIVE_OUT_REGISTER_LIST)
                tc_dir = "testcases"
            if not make_new_testcases or tc_gen_proc_rc == 0 :
                row = _redefine_live_out_indiv(row=row, live_out_register_list=live_out_register_list,
                            def_in_register_list=copy(DEF_IN_REGISTER_LIST), path_to_spurious_dir=path_to_spurious_dir,
                            spurious_program_list=spurious_program_list, path_to_disassembly_dir=path_to_disassembly_dir,
                            optimized_flag=optimized_flag, debug=debug, timeout=timeout, tc_directory=tc_dir)
                if row["opt_unopt_correctness"] == "no":
                    row = _redefine_live_out_indiv(row=row, live_out_register_list=copy(AMD64_ABI_REGISTERS_W_FP),
                            def_in_register_list=copy(DEF_IN_REGISTER_LIST), path_to_spurious_dir=path_to_spurious_dir,
                            spurious_program_list=spurious_program_list, path_to_disassembly_dir=path_to_disassembly_dir,
                            optimized_flag=optimized_flag, debug=debug, timeout=timeout, tc_directory=tc_dir)
                    if row["opt_unopt_correctness"] == "no":
                        row = _redefine_live_out_indiv(row=row, live_out_register_list=copy(AMD64_ABI_REGISTERS),
                                def_in_register_list=copy(DEF_IN_REGISTER_LIST), path_to_spurious_dir=path_to_spurious_dir,
                                spurious_program_list=spurious_program_list, path_to_disassembly_dir=path_to_disassembly_dir,
                                optimized_flag=optimized_flag, debug=debug, timeout=timeout, tc_directory=tc_dir)
        new_rows.append(row.to_dict())
        pbar.update()

    df_out = pd.DataFrame(new_rows)
    return df_out


REGISTER_LIST_REGEX = re.compile("(?<=({))[^}]+")


def register_list_from_string(register_list_string: str, pattern):
    match = pattern.search(register_list_string)
    registers = match.strip().split()
    return registers


def _tc_gen_symbolic_and_test_cost(row, path_to_disassembly_dir, new_tc_dir, bound, max_tcs, timeout):

    def_in = row["def_in"]
    live_out = row["live_out"]

    path_to_function = join(path_to_disassembly_dir, row["path_to_function"])
    fun_dir = function_path_to_functions_folder(path=path_to_function)

    function_basename = basename(path_to_function)
    function_name, _ = splitext(function_basename)

    tc_dir = join(path_to_function, new_tc_dir)
    if not os.path.exists(tc_dir):
        os.mkdir(tc_dir)
    tc_destination_path = join(tc_dir, f"{function_name}.tc")

    tc_gen_proc = _stoke_tcgen_symbolic_exec(path_to_function, tc_destination_path, fun_dir, def_in,
                                            live_out, max_tcs, bound, timeout)

    row[f"{row[new_tc_dir]}_stdout"] = tc_gen_proc.stdout
    row[f"{row[new_tc_dir]}_success"] = True if tc_gen_proc.returncode == 0 else False

    if tc_gen_proc.returncode == 0:
        row[new_tc_dir] = tc_destination_path

        def_in_register_list = register_list_from_string(row["def_in"], REGISTER_LIST_REGEX)
        live_out_register_list = register_list_from_string(row["live_out"], REGISTER_LIST_REGEX)

        stack_out = row["stack_out"]
        heap_out = row["heap_out"]

        # temporairly assert no stack out
        assert not stack_out

        cost_test_rc, cost_test_stdout, cost, correct = test_costfn(target_f=path_to_function,
                                                                    rewrite_f=path_to_function,
                                                                    testcases_f=tc_destination_path,
                                                                    fun_dir=fun_dir,
                                                                    def_in_refister_list=def_in_register_list,
                                                                    live_out_register_list=live_out_register_list,
                                                                    stack_out=stack_out,
                                                                    heap_out=heap_out,
                                                                    live_dangerously=True)
        assert correct == "yes"

    else:
        row[f"{row[new_tc_dir]}_success"] = True
        row["unopt_unopt_cost"] = np.nan

    return row, tc_gen_proc.returncode


def _stoke_tcgen_symbolic_exec(path_to_function: str, tc_destination_path: str, fun_dir: str, def_in: str, live_out: str,
                               max_tcs: int, bound: int, timeout: int):

    completed_process = subprocess.run(['stoke_tcgen', '--target', path_to_function, "--output", tc_destination_path,
                            '--functions', fun_dir, '--prune', '--max_tcs', str(max_tcs), '--bound', str(bound),
                            '--def_in', def_in, '--live_out', live_out, '--live_dangerously'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=timeout)

    return completed_process


def is_spurious_program(path_to_spurious_dir, spurious_program_list,
                         path_to_function, path_to_testcases, functions_dir,
                         def_in_register_list, live_out_register_list, heap_out, stack_out, timeout):

    for spurious_program in spurious_program_list:
        path_to_spurious_prog = join(path_to_spurious_dir, spurious_program)
        cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                              rewrite_f=path_to_spurious_prog,
                                                              testcases_f=path_to_testcases,
                                                              fun_dir=functions_dir,
                                                              def_in_register_list=def_in_register_list,
                                                              live_out_register_list=live_out_register_list,
                                                              heap_out=heap_out,
                                                              stack_out=stack_out,
                                                              timeout=timeout)
        if correct_str == "yes":
            return True
    return False




def test_costfn(target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str,
                def_in_register_list: List[str], live_out_register_list: List[str], stack_out: bool, heap_out: bool,
                live_dangerously: bool = True , timeout: int = 600):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ''
    try:
        if stack_out and heap_out:
            cost_test = subprocess.run(
            ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
            testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
            '{ 0 1 ... 255 }', '--cost', '100*correctness+measured+latency', "--heap_out", "--stack_out",
            '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem", 
            '--live_out', register_list_to_register_string(live_out_register_list)], 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        elif stack_out:
            cost_test = subprocess.run(
                ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
                 testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
                 '{ 0 1 ... 255 }', '--cost', '100*correctness+measured+latency', "--stack_out",
                 '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem",
                 '--live_out', register_list_to_register_string(live_out_register_list)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        elif heap_out:
            cost_test = subprocess.run(
                ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
                 testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
                 '{ 0 1 ... 255 }', '--cost', '100*correctness+measured+latency', "--heap_out",
                 '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem",
                 '--live_out', register_list_to_register_string(live_out_register_list)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        else: 
            cost_test = subprocess.run( ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, 
            '--testcases', testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
            '{ 0 1 ... 255 }', '--cost', '100*correctness+measured+latency',
            '--def_in', register_list_to_register_string(def_in_register_list), 
            '--live_out', register_list_to_register_string(live_out_register_list), ], 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if cost_test.returncode == 0:
            cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
            correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()
        else:
            #breakpoint()
            cost = -10701
            correct = "failed"
        return cost_test.returncode, cost_test.stdout, cost, correct

    except subprocess.TimeoutExpired as err:
        return -11785, err, -11785, "timeout"


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    if args.make_new_testcases:
        assert args.new_tc_dir, "if you're creating new testcases, you need to specify the directory to save them"

    df_in = pd.read_csv(args.path_to_stats_df)
    df_in = df_in[df_in["unopt_unopt_correctness"] == "yes"].reindex()

    if not args.debug:
        n_splits = 128
        df_length = int(len(df_in) / n_splits)
        frames = [df_in.iloc[i * df_length :(i + 1) * df_length].copy() for i in range(n_splits + 1)]
        jobs = []
        for i, frame in enumerate(frames):
            jobs.append({"path_to_disassembly_dir": args.path_to_disassembly_dir,
                         "df": deepcopy(frame),
                         "path_to_spurious_dir": args.path_to_spurious_dir,
                         "spurious_program_list": args.spurious_progs.split(":") if args.spurious_progs else None,
                         "optimized_flag": args.optimized_flag,
                         "position": (i%args.n_workers)+1,
                         "make_new_testcases": args.make_new_testcases,
                         "new_tc_dir": args.new_tc_dir,
                         "bound": args.bound,
                         "max_tcs": args.max_tcs})
        out_dfs = []
        pbar = tqdm(total=len(df_in), position=0, desc="all workers progress bar")
        for df in Pool(args.n_workers).imap(redefine_live_out_df_wrapper, jobs):
            out_dfs.append(df)
            pbar.update(len(df))
        df_out = pd.concat(out_dfs)

    else:
        df_out = redefine_live_out_df(path_to_disassembly_dir=args.path_to_disassembly_dir,
                                      df = df_in,
                                      path_to_spurious_dir=args.path_to_spurious_dir,
                                      spurious_program_list=args.spurious_progs.split(":") if args.spurious_progs else None,
                                      optimized_flag=args.optimized_flag,
                                      debug=True,
                                      timeout=args.timeout,
                                      make_new_testcases=args.make_new_testcases,
                                      new_tc_dir=args.new_tc_dir,
                                      bound=args.bound,
                                      max_tcs=args.max_tcs)

    df_out.to_csv(args.path_to_out_stats_df)

