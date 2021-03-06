import subprocess
import copy
import json
import threading
from logging import Logger
from typing import Dict
from typing import Union
import shutil
import os
from utils import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX, get_max_testcase_index, process_raw_assembly, \
    FUNCTION_BEGIN_REGEX, HACK_TEXT, replace_and_rewrite_rsp_loc, ALL_REGISTERS_LIVE_OUT
from time import time
import warnings
from os.path import join


def make_tunit_file(in_f: str, out_f: str, fun_dir: str, timeout: int = 100):
    try:
        with open(out_f, "w") as f:
            tunit_proc = subprocess.run(
                 ['/home/stoke/stoke/bin/stoke', 'debug', 'tunit', '--target', in_f,
                  '--functions', fun_dir, "--prune", "--live_dangerously"],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout)

    except subprocess.TimeoutExpired as err:
        return -11747, err

    return tunit_proc.returncode, tunit_proc.stdout


def test_costfn(target_f: str,
                rewrite_f: str,
                testcases_f: str,
                fun_dir: str,
                settings_conf: Dict[str, str],
                timeout: int = 100):
    try:
        if settings_conf.get("heap_out"):
            cost_test = subprocess.run(
                    ['/home/stoke/stoke/bin/stoke', 'debug', 'cost',
                     '--target', target_f,
                     '--rewrite', rewrite_f,
                     '--testcases', testcases_f,
                     '--functions', fun_dir,
                     "--prune",
                     "--def_in", settings_conf["def_in"],
                     "--live_out", settings_conf["live_out"],
                     "--distance", settings_conf["distance"],
                     "--misalign_penalty", str(settings_conf["misalign_penalty"]),
                     "--sig_penalty", settings_conf["sig_penalty"],
                     "--cost", settings_conf["costfn"],
                     "--training_set", settings_conf["training_set"],
                     "--live_dangerously", "--heap_out"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout)
        else:
            cost_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'cost',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--testcases', testcases_f,
                 '--functions', fun_dir,
                 "--prune",
                 "--def_in", settings_conf["def_in"],
                 "--live_out", settings_conf["live_out"],
                 "--distance", settings_conf["distance"],
                 "--misalign_penalty", str(settings_conf["misalign_penalty"]),
                 "--sig_penalty", settings_conf["sig_penalty"],
                 "--cost", settings_conf["costfn"],
                 "--training_set", settings_conf["training_set"],
                 "--live_dangerously"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout)
        if cost_test.returncode == 0:
            cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
            correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()
        else:
            cost = -10701
            correct = "failed"
        return cost_test.returncode, cost_test.stdout, cost, correct

    except subprocess.TimeoutExpired as err:
        return -11785, err, -11785, "timeout"


def verify_rewrite(target_f: str,
                   rewrite_f: str,
                   fun_dir: str,
                   settings_conf: Dict[str, str],
                   bound: int = 64,
                   testcases_f: str = None,
                   machine_output_f: str = "tmp.txt",
                   aliasing_strategy: str = "basic",
                   strategy: str = "bounded",
                   timeout: int = 60,
                   hack_validator: bool = False) -> (int, str):

    if hack_validator:
        assert strategy == "bounded", "in order to use 'hack_validator' setting, bounded strategy needs to be used"
        # change path for the orig target so you don't over-write or delete it 
        fun_dir = os.path.dirname(target_f)
        old_target_f = copy.copy(target_f)
        old_rewrite_f = copy.copy(rewrite_f)

        file_to_remove_1 = os.path.splitext(target_f)[0] + "_target.s"
        file_to_remove_2 = os.path.splitext(rewrite_f)[0] + "_rewrite.s"
        for f in (file_to_remove_1, file_to_remove_2):
            if os.path.exists(f) and open(f).read() == "":
                os.remove(f)
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        target_f = join("tmp", os.path.basename(target_f))
        rewrite_f = join("tmp", os.path.basename(rewrite_f))
        target_f = os.path.splitext(target_f)[0] + "_{}_target.s".format(time()+threading.get_ident())
        rewrite_f = os.path.splitext(rewrite_f)[0] + "_{}_rewrite.s".format(time()+threading.get_ident())

        rc_tgt_rc = read_write_assembly2_hacked(path_to_input=old_target_f, path_to_output=target_f, fun_dir = fun_dir, timeout=timeout)
        rc_rewrite_rc = read_write_assembly2_hacked(path_to_input=old_rewrite_f, path_to_output=rewrite_f, fun_dir = fun_dir, timeout=timeout)
        if not rc_tgt_rc == 0 and rc_rewrite_rc == 0:
            warnings.warn("function {} tunit for hacking failed".format(target_f))
            return -1, "tunit for hacking failed"
    #print(f"rewrite f is {rewrite_f}, inside it is\n\n{open(rewrite_f).read()}", flush=True)
    try:
        if settings_conf["heap_out"]:
            if strategy in ("bounded", "ddec"):
                verify_test = subprocess.run(
                    ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                     '--target', target_f,
                     '--rewrite', rewrite_f,
                     '--machine_output', machine_output_f,
                     '--strategy', strategy,
                     '--alias_strategy', aliasing_strategy,
                     '--functions', fun_dir,
                     "--prune", "--live_dangerously",
                     "--def_in", settings_conf["def_in"],
                     "--live_out", settings_conf["live_out"],
                     "--distance", settings_conf["distance"],
                     "--misalign_penalty", str(settings_conf["misalign_penalty"]), 
                     "--sig_penalty", settings_conf["sig_penalty"],
                     "--cost", settings_conf["costfn"],
                     "--bound", str(bound),
                     "--heap_out"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout)
            # if not using the validator, then use hold-out verficiation
            else:
                assert testcases_f, "need testcases to run hold out verification"
                verify_test = subprocess.run(
                    ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                     '--target', target_f,
                     '--rewrite', rewrite_f,
                     '--machine_output', machine_output_f,
                     '--strategy', strategy,
                     '--testcases', testcases_f,
                     '--functions', fun_dir,
                     "--prune", "--live_dangerously",
                     "--def_in", settings_conf["def_in"],
                     "--live_out", settings_conf["live_out"],
                     "--distance", settings_conf["distance"],
                     "--misalign_penalty", str(settings_conf["misalign_penalty"]),
                     "--sig_penalty", settings_conf["sig_penalty"],
                     "--cost", settings_conf["costfn"],
                     "--heap_out"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout)
        # if not using the heap / testing the heap
        else:
            if strategy in ("bounded", "ddec"):
                verify_test = subprocess.run(
                    ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                     '--target', target_f,
                     '--rewrite', rewrite_f,
                     '--machine_output', machine_output_f,
                     '--strategy', strategy,
                     '--alias_strategy', aliasing_strategy,
                     '--functions', fun_dir,
                     "--prune", "--live_dangerously",
                     "--def_in", settings_conf["def_in"],
                     "--live_out", settings_conf["live_out"],
                     "--distance", settings_conf["distance"],
                     "--misalign_penalty", str(settings_conf["misalign_penalty"]), 
                     "--sig_penalty", settings_conf["sig_penalty"],
                     "--cost", settings_conf["costfn"],
                     "--bound", str(bound)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout)
            else:
                # if not using the validator, then use hold-out verficiation
                assert testcases_f, "need testcases to run hold out verification"
                verify_test = subprocess.run(
                    ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                     '--target', target_f,
                     '--rewrite', rewrite_f,
                     '--machine_output', machine_output_f,
                     '--strategy', strategy,
                     '--testcases', testcases_f,
                     '--functions', fun_dir,
                     "--prune", "--live_dangerously",
                     "--def_in", settings_conf["def_in"],
                     "--live_out", settings_conf["live_out"],
                     "--distance", settings_conf["distance"],
                     "--misalign_penalty", str(settings_conf["misalign_penalty"]),
                     "--sig_penalty", settings_conf["sig_penalty"],
                     "--cost", settings_conf["costfn"]],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout)
        rc = verify_test.returncode; stdout = verify_test.stdout;
    except subprocess.TimeoutExpired as err:
        rc = -1; stdout = err;
    if hack_validator:
        os.remove(target_f)
        os.remove(rewrite_f)
    return rc, stdout


def parse_verify_machine_output(machine_output_f: str) -> (bool, bool, Union[None, str]):
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)

    verified_correct = machine_output_dict["verified"]
    counter_examples_available = machine_output_dict["counter_examples_available"]
    counterexample_str = machine_output_dict["counterexample"] if not verified_correct and counter_examples_available \
                            else None
    return verified_correct, counter_examples_available, counterexample_str


def add_counterexample_to_testcases(counterexample_str: str, path_to_testcases: str):
    with open(path_to_testcases, "r") as fh:
        tc_str = fh.read()
    max_tc_idx = get_max_testcase_index(tc_str)
    with open(path_to_testcases, "a") as fh:
        fh.write(f"\n\n\nTestcase {str(max_tc_idx)}\n\n:")
        fh.write(counterexample_str)


def verify_and_rewrite_testcase(container_path_to_target: str,
                                container_path_to_rewrite: str,
                                container_path_to_testcases: str,
                                container_path_to_functions: str,
                                container_path_to_machine_output: str,
                                settings_conf: Dict[str, str],
                                strategy: str = "hold_out",
                                alias_strategy: str = "basic",
                                bound: int = 2,
                                timeout: int = 300,
                                hack_validator: bool = False):

    verify_returncode, verify_stdout = verify_rewrite(target_f=container_path_to_target,
                                                      rewrite_f=container_path_to_rewrite,
                                                      fun_dir=container_path_to_functions,
                                                      machine_output_f=container_path_to_machine_output,
                                                      testcases_f=container_path_to_testcases,
                                                      strategy=strategy,
                                                      settings_conf=settings_conf,
                                                      bound=bound,
                                                      aliasing_strategy=alias_strategy,
                                                      timeout=timeout,
                                                      hack_validator=hack_validator)

    if verify_returncode == 0:
        verified_correct, counter_examples_available, counterexample_str = \
            parse_verify_machine_output(container_path_to_machine_output)

        if not verified_correct and counter_examples_available:
            add_counterexample_to_testcases(counterexample_str=counterexample_str,
                                            path_to_testcases=container_path_to_testcases)
    else:
        verified_correct = False
        counter_examples_available = False

    return verified_correct, counter_examples_available, verify_stdout


def _assembly2_hacked(input_assembly: str):
    # injects a string into the assembly such that it prevents %rsp from being aliased for memory clobbering
    assembly = process_raw_assembly(raw_assembly=input_assembly,
                                              preserve_fun_names=True,
                                              preserve_semantics=True)
    # split it only once, you split on the first occurrenct of a location (the prog start)
    m = FUNCTION_BEGIN_REGEX.search(assembly)
    if m:
        body_begin_idx = m.end()
        metadata = assembly[:body_begin_idx]
        body = assembly[body_begin_idx:]
    else:
        metadata = body = ""

    return metadata + HACK_TEXT + body


def read_write_assembly2_hacked(path_to_input: str, path_to_output: str, fun_dir: str, timeout: int = 100):

    tmp_raw_asm_path = os.path.splitext(path_to_output)[0] + ".raw"
#    fun_dir = os.path.dirname(path_to_input)

    raw_assembly = open(path_to_input).read()
    hacked_asm_string = _assembly2_hacked(input_assembly=raw_assembly)
    #print(f"hacked asm string looks like\n\n{hacked_asm_string}\n\nwhereas og one was\n\n{raw_assembly}", flush=True)

    with open(tmp_raw_asm_path, "w") as fh:
        fh.write(hacked_asm_string)

    tunit_rc, tunit_stdout = make_tunit_file(in_f=tmp_raw_asm_path,
                                             out_f=path_to_output,
                                             fun_dir=fun_dir,
                                             timeout=timeout)
    if tunit_rc == 0:
        replace_and_rewrite_rsp_loc(path_to_formatted_asm = path_to_output)

    os.remove(tmp_raw_asm_path)

    return tunit_rc



