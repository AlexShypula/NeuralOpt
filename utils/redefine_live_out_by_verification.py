import json
import subprocess
import os
import pandas as pd
from os.path import join
from stoke_preprocess import hash_file
from make_data import function_path_to_optimized_function, function_path_to_testcases, function_path_to_functions_folder,\
    remove_first_n_dirs
from typing import List, Dict, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from registers import LIVE_OUT_REGISTER_LIST
import re
from redefine_live_out import clean_ansi_color_codes, register_list_to_register_string, test_if_lower_order_register
from registers import DEF_IN_REGISTER_LIST, LIVE_OUT_REGISTER_LIST, REGISTER_TO_STDOUT_REGISTER, \
    GP_REGISTERS_SET, LIVE_OUT_FLAGS_SET, gp_reg_64_to_32, gp_reg_64_to_16, gp_reg_64_to_8, AMD64_ABI_REGISTERS_W_FP, \
    AMD64_ABI_REGISTERS
from multiprocessing.pool import ThreadPool
from redefine_live_out import test_costfn
from stoke_test_costfn import StopWatch
from copy import copy


DIFF_REGEX = re.compile("(?<=(Difference of running target and rewrite on the counterexample:))[\s\S]*")
LIVE_OUT_FLAGS_REGEX = re.compile("|".join(["({})".format(r) for r in LIVE_OUT_FLAGS_SET]))

@dataclass
class ParseOptions:
    path_to_disassembly_dir: str = field(metadata=dict(args=["--path_to_disassembly_dir"]))
    path_to_input_dataframe: str = field(metadata=dict(args=["--path_to_input_dataframe"]))
    path_to_output_dataframe: str = field(metadata=dict(args=["--path_to_output_dataframe"]))
    aliasing_strategy: str = field(metadata=dict(args=["--aliasing_strategy"]), default="basic")
    bound: int = field(metadata=dict(args=["--bound"]), default=2)
    verification_timeout: int = field(metadata=dict(args=["--verification_timeout"]), default=60)
    cost_timeout: int = field(metadata=dict(args=["--cost_timeout"]), default=300)
    n_threads: int = field(metadata=dict(args=["--n_threads"]), default = 8)
    debug: bool = field(metadata=dict(args=["--debug"]), default = False)
    testcases_dir: str = field(metadata=dict(args=["--testcases_dir"]), default="testcases")
    depth_of_testing: int = field(metadata=dict(args=["--depth_of_testing"]), default = 8)
   


def main(path_to_input_dataframe: str, path_to_output_dataframe: str, n_threads: int, path_to_disassembly_dir: str,
         bound: int, aliasing_strategy: str, verification_timeout: int, cost_timeout: int,  debug: bool = False,
         depth_of_testing: int = 5, **kwargs):

    in_df = pd.read_csv(path_to_input_dataframe)
    jobs = []
    for i, row in in_df.iterrows():
        jobs.append({"row": row, "path_to_disassembly_dir": path_to_disassembly_dir, "job_id": i, "bound": bound,
                     "debug": debug, "aliasing_strategy": aliasing_strategy, "verification_timeout": verification_timeout,
                     "cost_timeout": cost_timeout, "depth_of_testing": depth_of_testing})

    pbar = tqdm(total=len(jobs))

    out_df_list = []
    n_verified = 0
    if not debug:
        for row in ThreadPool(n_threads).imap_unordered(_process_training_example_with_redefine_verify_wrapper, jobs):
            n_verified+=row["verified_correct"]
            out_df_list.append(row.to_dict())
            pbar.set_description("verifying all assembly progress, {} have verified".format(n_verified))
            pbar.update()
    else:
        for row in map(_process_training_example_with_redefine_verify_wrapper, jobs):
            n_verified+=row["verified_correct"]
            out_df_list.append(row.to_dict())
            pbar.set_description("verifying all assembly progress, {} have verified".format(n_verified))
            pbar.update()
    print("a total of {} of {} verified for {:2f}% percent".format(n_verified, len(in_df), n_verified/len(in_df)))
    out_df = pd.DataFrame(out_df_list)
    out_df.to_csv(path_to_output_dataframe)


def verify_rewrite(target_f: str,
                rewrite_f: str,
                fun_dir: str,
                def_in: str,
                live_out: str,
                heap_out: bool,
                costfn: str,
                bound: int = 64,
                machine_output_f: str = "tmp.txt",
                aliasing_strategy: str = "basic",
                strategy: str = "bounded",
                timeout: int = 60) -> (int, str):

    try:
        if heap_out:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
                 '--alias_strategy', aliasing_strategy,
                 '--functions', fun_dir,
                 "--prune", "--live_dangerously",
                 "--def_in", def_in,
                 "--live_out", live_out,
                 "--distance", "hamming",
                 "--misalign_penalty", "1",
                 "--sig_penalty", "9999",
                 "--cost", costfn,
                 "--bound", str(bound),
                 "--heap_out"],
                stdout = subprocess.PIPE,stderr = subprocess.STDOUT,
                text = True, timeout = timeout)
        else:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
                 '--alias_strategy', aliasing_strategy,
                 '--functions', fun_dir,
                 "--prune", "--live_dangerously",
                 "--def_in", def_in,
                 "--live_out", live_out,
                 "--distance", "hamming",
                 "--misalign_penalty", "1",
                 "--sig_penalty", "9999",
                 "--cost", costfn,
                 "--bound", str(bound)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout)
        return verify_test.returncode, verify_test.stdout
    except subprocess.TimeoutExpired as err:
        return -1, f"verify timed out with error {err}"


def parse_verify_machine_output(machine_output_f: str) -> bool:
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)
    verified_correct = machine_output_dict["verified"]
    counter_example_avail = machine_output_dict["counter_examples_available"]
    return verified_correct, counter_example_avail


def verify_and_parse(target_f: str,
                        rewrite_f: str,
                        fun_dir: str,
                        def_in: str,
                        live_out: str,
                        heap_out: bool,
                        costfn: str,
                        bound: int = 64,
                        machine_output_f: str = "tmp.txt",
                        aliasing_strategy: str = "bounded",
                        strategy: str = "bounded",
                        timeout: int = 60):

    verify_returncode, verify_stdout =  verify_rewrite(target_f=target_f,
                                        rewrite_f=rewrite_f,
                                        fun_dir=fun_dir,
                                        def_in=def_in,
                                        live_out=live_out,
                                        heap_out=heap_out,
                                        costfn=costfn,
                                        bound=bound,
                                        machine_output_f=machine_output_f,
                                        aliasing_strategy=aliasing_strategy,
                                        strategy=strategy,
                                        timeout=timeout)
    if verify_returncode == 0:
        verified_correct, counter_examples_avail = parse_verify_machine_output(machine_output_f)
    else:
        verified_correct = False
        counter_examples_avail = False

    return verify_returncode, verified_correct, counter_examples_avail, clean_ansi_color_codes(verify_stdout)


def verify_and_parse_with_diff(**kwargs):

    verify_returncode, verified_correct, counter_examples_avail, verify_stdout = verify_and_parse(**kwargs)

    diff_str = None
    if not verified_correct and counter_examples_avail:
        if m := DIFF_REGEX.search(verify_stdout):
            diff_str = m.group()

    return verify_returncode, verified_correct, counter_examples_avail, verify_stdout, diff_str

def _stoke_redefine_regs_verification(def_in_register_list: List[str], live_out_register_list: List[str],
                               target_f: str, rewrite_f: str, fun_dir: str, heap_out: bool,
                               cost_fn: str, machine_output_f: str, depth_of_testing: int, aliasing_strategy: str = "basic",
                               strategy: str = "bounded", bound: int = 64, live_dangerously: bool = False,
                               debug: bool = False, timeout: int = 60):
    #breakpoint()

    def_in_str = register_list_to_register_string(def_in_register_list)
    live_out_str = register_list_to_register_string(live_out_register_list)

    verify_returncode, verified_correct, counter_examples_avail, verify_stdout, diff_str = verify_and_parse_with_diff(target_f=target_f,
                                                                           rewrite_f=rewrite_f,
                                                                           fun_dir=fun_dir,
                                                                           def_in=def_in_str,
                                                                           live_out=live_out_str,
                                                                           heap_out=heap_out,
                                                                           costfn=cost_fn,
                                                                           bound=bound,
                                                                           machine_output_f=machine_output_f,
                                                                           aliasing_strategy=aliasing_strategy,
                                                                           strategy=strategy,
                                                                           timeout=timeout
                                                                           )

    if verified_correct or verify_returncode != 0 or not counter_examples_avail:
        if debug: 
            print("target file is {} and current depth is".format(target_f, depth_of_testing))
            print("verified is: {}".format(verified_correct))
            print("verified rc: {}".format(verify_returncode))
            print("counter example avail: {}".format(counter_examples_avail))
            print("def_in_register_list: " + " ".join(def_in_register_list)) 
            print("live out register list: " + " ".join(live_out_register_list))
            print("orig live_out_register_list: " + " ".join(def_in_register_list))
            print("diff std out cleaned" + diff_str if diff_str else "no diff avail" )
            print("verified std out is:\n" + verify_stdout, flush=True) 
       
 
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, depth_of_testing

    else:
        has_diff_registers = False
        new_live_out_register_list = []
        for register in live_out_register_list:
            register_stdout_code = REGISTER_TO_STDOUT_REGISTER[register]
            if type(diff_str) != str: 
                breakpoint()
            findall_string = "(?<=(?:{}))[^\n]+".format(register_stdout_code)
            findall_result = re.findall(findall_string, diff_str)
            has_diff_registers |= (findall_result != [])
            # if there is no regular expressions match (i.e. this register is equal in unopt and opt versions)
            # findall will return an empty list
            if findall_result == []:
                new_live_out_register_list.append(register)
            # only if it is a general purpose register will we try to search for a lower register to test out
            elif register in GP_REGISTERS_SET:
                assert len(findall_result) == 2, "findall result is not length 2, result is {} and diff is {}" \
                    .format(findall_result, diff_str)
                lower_order_register_result = test_if_lower_order_register(match_list=findall_result,
                                                                           gp_register_name_64=register)
                # the test returns a register name if a lower-order register is found, otherwise None
                if lower_order_register_result != None:
                    new_live_out_register_list.append(lower_order_register_result)
        if debug:
            print("target file is {} and current depth is".format(target_f, depth_of_testing))
            print("def_in_register_list: " + " ".join(def_in_register_list))
            print("orig live_out_register_list: " + " ".join(def_in_register_list))
            print("diff std out cleaned" + diff_str)
            print("verified is: {}".format(verified_correct))
            print("new_live_out_list: " + " ".join(new_live_out_register_list), flush=True)

        # if any flags are in the diff, remove all flags (i.e. don't test FLAGS register)
        if LIVE_OUT_FLAGS_REGEX.search(diff_str):
            new_live_out_register_list = [r for r in new_live_out_register_list if r not in LIVE_OUT_FLAGS_SET]
            has_diff_registers = True

        # recursivley call to try again with a new set of live_out
        if has_diff_registers and ((depth_of_testing := depth_of_testing-1) != 0):
            return _stoke_redefine_regs_verification(def_in_register_list, new_live_out_register_list,
                           target_f, rewrite_f, fun_dir, heap_out,
                           cost_fn, machine_output_f, depth_of_testing, aliasing_strategy, strategy,
                           bound, live_dangerously, debug, timeout)

        else:
            return verify_returncode, verified_correct, verify_stdout, diff_str, new_live_out_register_list, depth_of_testing


REGISTER_LIST_REGEX = re.compile("(?<=({))[^}]+")


def register_list_from_regex(stoke_diff_string: str, pattern: re.Pattern):
    match = pattern.search(stoke_diff_string)
    string = match.group()
    register_list = string.strip().split(" ")
    return register_list


def _stoke_redefine_live_out_verification(target_f: str, rewrite_f: str, fun_dir: str, def_in_str: str,
                                           machine_output_f: str, live_out_str: str, cost_fn: str,
                                           bound: int, depth_of_testing: int, aliasing_strategy: str = "basic",
                                           strategy="bounded", live_dangerously: bool = True, debug: bool = False,
                                           timeout: int = 60):

    def_in_register_list = register_list_from_regex(def_in_str, REGISTER_LIST_REGEX)
    live_out_register_list = register_list_from_regex(live_out_str, REGISTER_LIST_REGEX)

    heap_out = False
    verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, depth_of_testing =  \
        _stoke_redefine_regs_verification(def_in_register_list, live_out_register_list, target_f, rewrite_f, fun_dir,
                                          heap_out, cost_fn, machine_output_f, depth_of_testing, aliasing_strategy,
                                          strategy, bound, live_dangerously, debug, timeout)

    if verify_returncode != 0 or not verified_correct:
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, heap_out, depth_of_testing

    else:
        # we only test on this branch if it previously verified correct while we had heap_out set to false
        # this will let us know if we can be more conservative and enable memory or not
        new_heap_out = True
        def_in_str = register_list_to_register_string(def_in_register_list)
        live_out_str = register_list_to_register_string(live_out_register_list)
        new_verify_returncode, new_verified_correct, counter_examples_avail, new_verify_stdout, new_diff_str = \
            verify_and_parse_with_diff(target_f=target_f,
                                       rewrite_f=rewrite_f,
                                       fun_dir=fun_dir,
                                       def_in=def_in_str,
                                       live_out=live_out_str,
                                       heap_out=new_heap_out,
                                       costfn=cost_fn,
                                       bound=bound,
                                       machine_output_f=machine_output_f,
                                       aliasing_strategy=aliasing_strategy,
                                       strategy=strategy,
                                       timeout=timeout, 
                                       )
        if new_verified_correct:
            heap_out = new_heap_out
            verify_returncode = new_verify_returncode
            verified_correct = new_verified_correct
            verify_stdout = new_verify_stdout
            diff_str = new_diff_str

        os.remove(machine_output_f)
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, heap_out, depth_of_testing


def _process_training_example_with_redefine_verify(row: pd.Series, path_to_disassembly_dir: str, job_id: int, bound: int,
                                                   aliasing_strategy: str, debug: bool, testcases_dir: str = "testcases",
                                                   verification_timeout: int = 60, cost_timeout: int = 300, depth_of_testing: int = 5):

    performance_timer = StopWatch()
    performance_timer.start()
    performance_timer.new_event("validation_time")
    performance_timer.new_event("cost_time")

    target_f = remove_first_n_dirs(row["path_to_function"], 1)
    target_f = join(path_to_disassembly_dir, target_f)
    fun_dir = function_path_to_functions_folder(path=target_f)

    rewrite_f = function_path_to_optimized_function(target_f, optimized_flag="Og")

    def_in = row["def_in"]
    live_out = register_list_to_register_string(copy(LIVE_OUT_REGISTER_LIST))
    # live_out = row["live_out"]
    costfn = "100*corectness+measured+latency" # row["costfn"]
    performance_timer.validation_time.start()
    verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, heap_out, depth_of_testing = \
        _stoke_redefine_live_out_verification(target_f=target_f,
                                               rewrite_f=rewrite_f,
                                               fun_dir=fun_dir,
                                               def_in_str=def_in,
                                               live_out_str=live_out,
                                               cost_fn=costfn,
                                               bound=bound,
                                               aliasing_strategy=aliasing_strategy,
                                               machine_output_f="tmp_{}.txt".format(job_id),
                                               strategy="bounded",
                                               debug=debug,
                                               timeout=verification_timeout,
                                               depth_of_testing = depth_of_testing
                                               )
    performance_timer.validation_time.stop()
    row["live_out"] = register_list_to_register_string(live_out_register_list)
    row["validation_heap_out"] = heap_out
    row["depth_of_testing"] = depth_of_testing
    row["verification_returncode"] = verify_returncode
    row["verified_correct"] = verified_correct
    row["verify_stdout"] = verify_stdout
    row["verify_diff_str"] = diff_str
    row["cost_time"] = performance_timer.validation_time.time

    if verified_correct:
        performance_timer.cost_time.start()
        testcases_f = function_path_to_testcases(path=target_f, tc_folder=testcases_dir)
        cost_rc, cost_stdout, cost, correctness = test_costfn(target_f=target_f, rewrite_f=rewrite_f,
                                                              testcases_f=testcases_f,fun_dir=fun_dir,
                                                              def_in_register_list=register_list_from_regex(def_in, REGISTER_LIST_REGEX),
                                                              live_out_register_list=live_out_register_list, stack_out=False,
                                                              heap_out=heap_out, live_dangerously=True, timeout=cost_timeout)
        performance_timer.cost_time.stop()
        row["verified_cost"] = cost
        row["verified_cost_correctness"] = correctness
        row["verified_cost_rc"] = cost_rc
        row["verified_cost_stdout"] = cost_stdout
        row["validation_time"] = performance_timer.cost_time.time

    return row


def _process_training_example_with_redefine_verify_wrapper(kwags):
    return _process_training_example_with_redefine_verify(**kwags)


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    main(**vars(args))
