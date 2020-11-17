import subprocess
import re
import pandas as pd
from copy import copy, deepcopy
from make_data import function_path_to_optimized_function, function_path_to_testcases, function_path_to_functions_folder
from typing import List
from stoke_test_costfn import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from os.path import join
from tqdm import tqdm
from multiprocessing import Pool
from registers import DEF_IN_REGISTER_LIST, LIVE_OUT_REGISTER_LIST, REGISTER_TO_STDOUT_REGISTER, \
    GP_REGISTERS_SET, LIVE_OUT_FLAGS_SET, gp_reg_64_to_32, gp_reg_64_to_16, gp_reg_64_to_8


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

def redefine_live_out_df(path_to_disassembly_dir: str, df: pd.DataFrame, path_to_spurious_dir, spurious_program_list,
                         optimized_flag = "Og", position: int = 0, debug: bool = False):
    new_rows = []
    pbar = tqdm(total = len(df), position = position)
    cts = 0
    for i, row in df.iterrows():
        pbar.update()
        def_in_register_list = copy(DEF_IN_REGISTER_LIST) # not nested; so we can use copy
        live_out_register_list = copy(LIVE_OUT_REGISTER_LIST) # not nested; so we can use copy
        if row["unopt_unopt_correctness"] == "yes":

            path_to_function = join(path_to_disassembly_dir,  row["path_to_function"])
            #print("path to function is ", path_to_function)

            functions_dir = function_path_to_functions_folder(path=path_to_function)
            path_to_optimized_function = function_path_to_optimized_function(path = path_to_function,
                                                                             optimized_flag = optimized_flag)
            #print("path to rewrite is", path_to_optimized_function)
            path_to_testcases = function_path_to_testcases(path = path_to_function, tc_folder = "testcases")
            #print("path to testcases is ", path_to_testcases)
            # iteratively re-define live-out
            diff_rc, diff_stdout, live_out_register_list = stoke_diff_get_live_out_v2(def_in_register_list=def_in_register_list,
                                                                                   live_out_register_list=live_out_register_list,
                                                                                   target_f=path_to_function,
                                                                                   rewrite_f=path_to_optimized_function,
                                                                                   testcases_f=path_to_testcases,
                                                                                   fun_dir = functions_dir,
                                                                                   live_dangerously=True,
                                                                                   debug=debug)
            #print("diff rc is", diff_rc)
            #if diff_rc != 0:
                #print(diff_stdout)
                #print("path to rewrite is ", path_to_optimized_function)

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
                                                                     stack_out=False)
                if correct_str == "yes":
                    if path_to_spurious_dir and spurious_program_list:
                        is_spurious = is_spurious_program(path_to_spurious_dir=path_to_spurious_dir,
                                                           spurious_program_list=spurious_program_list,
                                                           path_to_function=path_to_function,
                                                           path_to_testcases=path_to_testcases,
                                                           functions_dir=functions_dir,
                                                           def_in_register_list=def_in_register_list,
                                                           live_out_register_list=live_out_register_list,
                                                           heap_out=True, stack_out=False)
                        correct_str = "spurious" if is_spurious else correct_str
                    row["heap_out"] = True
                    row["stack_out"] = False
                    row["opt_unopt_cost"] = float(cost)
                    row["opt_unopt_correctness"] = correct_str
                    row["def_in"] = register_list_to_register_string(def_in_register_list)
                    row["live_out"] = register_list_to_register_string(live_out_register_list)
                    row["opt_cost_str"] = cost_stdout
                    new_rows.append(row.to_dict())
                    #print("correct cost and appended")
                # try to then suppress the stack
                elif correct_str == "no":
                    cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                                          rewrite_f=path_to_optimized_function,
                                                                          testcases_f=path_to_testcases,
                                                                          fun_dir=functions_dir,
                                                                          def_in_register_list=def_in_register_list,
                                                                          live_out_register_list=live_out_register_list,
                                                                          heap_out=False,
                                                                          stack_out=False)

                    if correct_str == "yes":
                        if path_to_spurious_dir and spurious_program_list:
                            is_spurious = is_spurious_program(path_to_spurious_dir=path_to_spurious_dir,
                                                           spurious_program_list=spurious_program_list,
                                                           path_to_function=path_to_function,
                                                           path_to_testcases=path_to_testcases,
                                                           functions_dir=functions_dir,
                                                           def_in_register_list=def_in_register_list,
                                                           live_out_register_list=live_out_register_list,
                                                           heap_out=False, stack_out=False)
                            correct_str = "spurious" if is_spurious else correct_str
                        row["heap_out"] = False
                        row["stack_out"] = False
                        row["opt_unopt_cost"] = float(cost)
                        row["opt_unopt_correctness"] = correct_str
                        row["def_in"] = register_list_to_register_string(def_in_register_list)
                        row["live_out"] = register_list_to_register_string(live_out_register_list)
                        row["opt_cost_str"] = cost_stdout
                        new_rows.append(row.to_dict())
                        #print("cost run second time and correct")

                    # test only the registers
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
                                timeout=25
                            )
                            print(f"even after redefine live out, it was incorrect, live out is "
                                  f"{register_list_to_register_string(live_out_register_list)}")
                            print(f"diff stdout is {diff.stdout}")
                            #pass
                        #breakpoint()
                    # regardless of the outcome, add the result for progress bar tracking
                    row["opt_unopt_cost"] = float(cost)
                    row["opt_unopt_correctness"] = correct_str
                    row["def_in"] = register_list_to_register_string(def_in_register_list)
                    row["live_out"] = register_list_to_register_string(live_out_register_list)
                    row["diff_str"] = diff_stdout
                    row["opt_cost_str"] = cost_stdout
                    new_rows.append(row.to_dict())
                    #print("cost run second time and failed")

                # if correct_str from test_costfn is neither "yes" nor "no"
                else:
                    if debug:
                        pass
                        #breakpoint()
                    row["opt_unopt_correctness"] = correct_str
                    new_rows.append(row.to_dict())
                    #raise ValueError(f"correct str needs to be 'yes' or 'no' it was {correct_str}")
            # if diff process did not return 0
            else:
                if debug:
                    pass
                    #breakpoint()
                row["def_in"] = register_list_to_register_string(def_in_register_list)
                row["live_out"] = register_list_to_register_string(live_out_register_list)
                new_rows.append(row.to_dict())
                continue
        #pbar.update()
    df_out = pd.DataFrame(new_rows) 
    #print("number of unprocessed cts is ", cts)
    return df_out

def is_spurious_program(path_to_spurious_dir, spurious_program_list,
                         path_to_function, path_to_testcases, functions_dir,
                         def_in_register_list, live_out_register_list, heap_out, stack_out):

    for spurious_program in spurious_program_list:
        path_to_spurious_prog = join(path_to_spurious_dir, spurious_program)
        cost_rc, cost_stdout, cost, correct_str = test_costfn(target_f=path_to_function,
                                                              rewrite_f=path_to_spurious_prog,
                                                              testcases_f=path_to_testcases,
                                                              fun_dir=functions_dir,
                                                              def_in_register_list=def_in_register_list,
                                                              live_out_register_list=live_out_register_list,
                                                              heap_out=heap_out,
                                                              stack_out=stack_out)
        if correct_str == "yes":
            return True
    return False




def test_costfn(target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str,
                def_in_register_list: List[str], live_out_register_list: List[str], stack_out: bool, heap_out: bool,
                live_dangerously: bool = True ):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ''
    try:
        if stack_out and heap_out:
            cost_test = subprocess.run(
            ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
            testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
            '{ 0 1 ... 31 }', '--cost', '100*correctness+measured+latency', "--heap_out", "--stack_out",
            '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem", 
            '--live_out', register_list_to_register_string(live_out_register_list)], 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25)
        elif stack_out:
            cost_test = subprocess.run(
                ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
                 testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
                 '{ 0 1 ... 31 }', '--cost', '100*correctness+measured+latency', "--stack_out",
                 '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem",
                 '--live_out', register_list_to_register_string(live_out_register_list)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25)
        elif heap_out:
            cost_test = subprocess.run(
                ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
                 testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
                 '{ 0 1 ... 31 }', '--cost', '100*correctness+measured+latency', "--heap_out",
                 '--def_in', register_list_to_register_string(def_in_register_list), "--relax_mem",
                 '--live_out', register_list_to_register_string(live_out_register_list)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25)
        else: 
            cost_test = subprocess.run( ['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, 
            '--testcases', testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set',
            '{ 0 1 ... 31 }', '--cost', '100*correctness+measured+latency',  
            '--def_in', register_list_to_register_string(def_in_register_list), 
            '--live_out', register_list_to_register_string(live_out_register_list), ], 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25)
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
    df_in = pd.read_csv(args.path_to_stats_df)

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
                         "position": (i%args.n_workers)+1})
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
                                      debug=True)

    df_out.to_csv(args.path_to_out_stats_df)

