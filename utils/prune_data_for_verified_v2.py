import json
import subprocess
import os
from os.path import join
from stoke_preprocess import hash_file
from make_data import function_path_to_optimized_function, function_path_to_functions_folder
from typing import List, Dict, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import re
from redefine_live_out import clean_ansi_color_codes, register_list_to_register_string, test_if_lower_order_register
from registers import DEF_IN_REGISTER_LIST, LIVE_OUT_REGISTER_LIST, REGISTER_TO_STDOUT_REGISTER, \
    GP_REGISTERS_SET, LIVE_OUT_FLAGS_SET, gp_reg_64_to_32, gp_reg_64_to_16, gp_reg_64_to_8, AMD64_ABI_REGISTERS_W_FP, \
    AMD64_ABI_REGISTERS
from multiprocessing.pool import ThreadPool

DIFF_REGEX = re.compile("(?<=(Difference of running target and rewrite on the counterexample:))[\s\S]*")
LIVE_OUT_FLAGS_REGEX = re.compile("|".join(["({})".format(r) for r in LIVE_OUT_FLAGS_SET]))

@dataclass
class ParseOptions:
    path_to_data_files_dir: str = field(metadata=dict(args=["-path_to_data_dir", "--path_to_data_dir"]))
    path_to_hash2metadata: str = field(metadata=dict(args=["-path_to_hash2metadata", "--path_to_hash2metadata"]))
    path_to_new_hash2metadata: str = field(metadata=dict(args=["-path_to_new_hash2metadata",
                                                           "--path_to_new_hash2metadata"]))
    asm_path_prefix: str = field(metadata=dict(args=["-asm_path_prefix", "--asm_path_prefix"]))
    path_to_error_log: str = field(metadata=dict(args=["-path_to_error_log", "--path_to_error_log"]))
    in_file_prefix: str = field(metadata=dict(args=["-in_file_prefix", "--in_file_prefix"]), default = "val")
    out_file_prefix: str = field(metadata=dict(args=["-out_file_prefix", "--out_file_prefix"]),
                                 default = "val_verified")
    src_file_suffix: str = field(metadata=dict(args=["-src_file_suffix", "--src_file_suffix"]), default = "src")
    tgt_file_suffix: str = field(metadata=dict(args=["-tgt_file_suffix", "--tgt_file_suffix"]), default = "tgt")
    n_threads: int = field(metadata=dict(args=["-n_thread", "--n_threads"]), default = 8)


def main(path_to_data_files_dir: str, path_to_hash2metadata: str, path_to_new_hash2metadata: str, in_file_prefix: str,
         out_file_prefix: str, asm_path_prefix: str, src_file_suffix: str, tgt_file_suffix: str, path_to_error_log: str,
         n_threads: int):
    hash2metadata = json.load(open(path_to_hash2metadata))
    n_verified = 0
    src_lines = open(join(path_to_data_files_dir, in_file_prefix + "." + src_file_suffix)).readlines()
    tgt_lines = open(join(path_to_data_files_dir, in_file_prefix + "." + tgt_file_suffix)).readlines()
    jobs = []
    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        asm_hash = hash_file(src_line.strip())
        metadata = hash2metadata[asm_hash]
        jobs.append({"src_string": src_line.strip(), "tgt_string": tgt_line, "asm_path_prefix": asm_path_prefix,
                     "metadata": metadata, "job_id": i})
    total_cts = len(src_lines)
    pbar = tqdm(total=total_cts)

    new_hash2metadata = {}
    with open(path_to_error_log, "w+") as err_log_fh,  \
            open(join(path_to_data_files_dir, out_file_prefix + "." + src_file_suffix), "w+") as out_src_fh, \
            open(join(path_to_data_files_dir, out_file_prefix + "." + tgt_file_suffix), "w+") as out_tgt_fh:

        for verified_correct, verify_stdout, diff_str, src_string, tgt_string, metadata in \
                ThreadPool(n_threads).imap_unordered(__process_training_example_with_redefine_verify_wrapper, jobs):
            if verified_correct:
                out_src_fh.write(src_string)
                out_tgt_fh.write(tgt_string)
                new_hash2metadata[hash_file(src_string.strip())] = metadata
                n_verified+=1
            else:
                err_log_fh.write("function {} didn't verify\n\n".format(metadata["name"]))
                err_log_fh.write("{}\n\n".format(verify_stdout))
                err_log_fh.write("Diff String Parsed was\n\n")
                err_log_fh.write("{}\n\n".format(diff_str))
                pbar.update()
            pbar.set_description("verifying all assembly progress, {} have verified".format(n_verified))
    print("a total of {} of {} verified for {:2f}% percent".format(n_verified, total_cts, n_verified/total_cts))
    with open(path_to_hash2metadata) as fh:
        json.dump(new_hash2metadata, fh, indent=4)


def verify_rewrite(target_f: str,
                rewrite_f: str,
                fun_dir: str,
                def_in: str,
                live_out: str,
                heap_out: bool,
                costfn: str,
                bound: int = 64,
                machine_output_f: str = "tmp.txt",
                strategy: str = "bounded") -> (int, str):
    try:
        if heap_out:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
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
                text = True, timeout = 30)
        else:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
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
                text=True, timeout=30)
        return verify_test.returncode, verify_test.stdout
    except subprocess.TimeoutExpired as err:
        return -1, f"verify timed out with error {err}"


def parse_verify_machine_output(machine_output_f: str) -> bool:
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)
    verified_correct = machine_output_dict["verified"]
    return verified_correct


def verify_and_parse(target_f: str,
                        rewrite_f: str,
                        fun_dir: str,
                        def_in: str,
                        live_out: str,
                        heap_out: bool,
                        costfn: str,
                        bound: int = 64,
                        machine_output_f: str = "tmp.txt",
                        strategy: str = "bounded"):

    verify_returncode, verify_stdout =  verify_rewrite(target_f=target_f,
                                        rewrite_f=rewrite_f,
                                        fun_dir=fun_dir,
                                        def_in=def_in,
                                        live_out=live_out,
                                        heap_out=heap_out,
                                        costfn=costfn,
                                        bound=bound,
                                        machine_output_f=machine_output_f,
                                        strategy=strategy)
    if verify_returncode == 0:
        verified_correct = parse_verify_machine_output(machine_output_f)
    else:
        verified_correct = False

    return verify_returncode, verified_correct, clean_ansi_color_codes(verify_stdout)


def verify_and_parse_with_diff(**kwargs):

    verify_returncode, verified_correct, verify_stdout = verify_and_parse(**kwargs)

    diff_str = None
    if not verified_correct:
        if m := DIFF_REGEX.search():
            diff_str = m.group()

    return verify_returncode, verified_correct, verify_stdout, diff_str


def _stoke_redefine_regs_verification(def_in_register_list: List[str], live_out_register_list: List[str],
                               target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, heap_out: bool,
                               cost_fn: str, machine_output_f: str, depth_of_testing: int, strategy: str = "bounded",
                               bound: int = 64, live_dangerously: bool = False, debug: bool = False, timeout: int = 600):

    def_in_str = register_list_to_register_string(def_in_register_list)
    live_out_str = register_list_to_register_string(live_out_register_list)


    verify_returncode, verified_correct, verify_stdout, diff_str = verify_and_parse_with_diff(target_f=target_f,
                                                                           rewrite_f=rewrite_f,
                                                                           fun_dir=fun_dir,
                                                                           def_in=def_in_str,
                                                                           live_out_str=live_out_str,
                                                                           heap_out=heap_out,
                                                                           cost_fn=cost_fn,
                                                                           bound=bound,
                                                                           machine_output_f=machine_output_f,
                                                                           strategy=strategy
                                                                           )



    if verified_correct or verify_returncode != 0:
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, depth_of_testing

    else:
        has_diff_registers = False
        new_live_out_register_list = []
        for register in live_out_register_list:
            register_stdout_code = REGISTER_TO_STDOUT_REGISTER[register]
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
            print("def_in_register_list: " + " ".join(def_in_register_list))
            print("orig live_out_register_list: " + " ".join(def_in_register_list))
            print("diff std out cleaned" + diff_str)
            print("new_live_out_list: " + " ".join(new_live_out_register_list))

        # if any flags are in the diff, remove all flags (i.e. don't test FLAGS register)
        if LIVE_OUT_FLAGS_REGEX.search(diff_str):
            new_live_out_register_list = [r for r in new_live_out_register_list if r not in LIVE_OUT_FLAGS_SET]
            has_diff_registers = True

        # recursivley call to try again with a new set of live_out
        if has_diff_registers and ((depth_of_testing := depth_of_testing-1) != 0):
            return _stoke_redefine_regs_verification(def_in_register_list, new_live_out_register_list,
                           target_f, rewrite_f, testcases_f, fun_dir, heap_out,
                           cost_fn, machine_output_f, depth_of_testing, strategy,
                           bound, live_dangerously, debug, timeout)

        else:
            return verify_returncode, verified_correct, verify_stdout, diff_str, new_live_out_register_list, depth_of_testing


def _stoke_redefine_live_out_verification(def_in_register_list: List[str], live_out_register_list: List[str],
                               target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, heap_out: bool,
                               cost_fn: str, machine_output_f: str, depth_of_testing: int, strategy: str = "bounded",
                               bound: int = 64, live_dangerously: bool = False, debug: bool = False, timeout: int = 600):

    verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, depth_of_testing =  \
        _stoke_redefine_regs_verification(def_in_register_list, live_out_register_list,
                           target_f, rewrite_f, testcases_f, fun_dir, heap_out,
                           cost_fn, machine_output_f, depth_of_testing, strategy,
                           bound, live_dangerously, debug, timeout)

    if verify_returncode != 0 or verified_correct or heap_out == False:
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, heap_out, depth_of_testing

    else:
        # we only test on this branch if it previously didn't verify AND heap_out was true, so we test with heap_out as
        # false now
        heap_out = False
        def_in_str = register_list_to_register_string(def_in_register_list)
        live_out_str = register_list_to_register_string(live_out_register_list)
        verify_returncode, verified_correct, verify_stdout, diff_str = \
            verify_and_parse_with_diff(target_f=target_f,
                                       rewrite_f=rewrite_f,
                                       fun_dir=fun_dir,
                                       def_in=def_in_str,
                                       live_out_str=live_out_str,
                                       heap_out=heap_out,
                                       cost_fn=cost_fn,
                                       bound=bound,
                                       machine_output_f=machine_output_f,
                                       strategy=strategy
                                       )
        os.remove(machine_output_f)
        return verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list, heap_out, depth_of_testing


def _process_training_example_with_redefine_verify(src_string: str, tgt_string: str, asm_path_prefix: str, metadata: Dict,
                                                   job_id: int):
    target_f = join(asm_path_prefix, metadata["base_asbly_path"])
    rewrite_f = function_path_to_optimized_function(target_f, optimized_flag="Og")
    fun_dir = function_path_to_functions_folder(target_f)
    def_in = metadata["def_in"]
    live_out = metadata["live_out"]
    heap_out = metadata["heap_out"]
    costfn = metadata["costfn"]
    bound = 64

    verify_returncode, verified_correct, verify_stdout, diff_str, live_out_register_list = \
        _stoke_redefine_live_out_verification(target_f=target_f,
                                               rewrite_f=rewrite_f,
                                               fun_dir=fun_dir,
                                               def_in=def_in,
                                               live_out_str=live_out,
                                               heap_out=heap_out,
                                               cost_fn=costfn,
                                               bound=bound,
                                               machine_output_f="tmp_{}.txt".format(job_id),
                                               strategy="bounded"
                                               )
    metadata["live_out"] = register_list_to_register_string(live_out_register_list)

    return verified_correct, verify_stdout, diff_str, src_string, tgt_string, metadata


def __process_training_example_with_redefine_verify_wrapper(kwags):
    return _process_training_example_with_redefine_verify(**kwags)




if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    main(**args)
