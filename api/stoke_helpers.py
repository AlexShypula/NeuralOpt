import subprocess
import json
from logging import Logger
from typing import Dict
from typing import Union
from utils import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX, get_max_testcase_index

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
                   timeout: int = 60) -> (int, str):

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
        return verify_test.returncode, verify_test.stdout
    except subprocess.TimeoutExpired as err:
        return -1, f"verify timed out with error {err}"


def parse_verify_machine_output(machine_output_f: str) -> (bool, bool, Union[None, str]):
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)

    verified_correct = machine_output_dict["verified"]
    counter_examples_available = machine_output_dict["counter_examples_available"]
    counterexample_str = machine_output_dict["counterexample"] if not verified_correct and counter_examples_available \
                            else None
    return verified_correct, counter_examples_available, counterexample_str


def add_counterexample_to_testcases(counterexample_str: str, path_to_testcases: str):
    with open(path_to_testcases, "a") as fh:
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
                                timeout: int = 300):

    verify_returncode, verify_stdout = verify_rewrite(target_f=container_path_to_target,
                                                      rewrite_f=container_path_to_rewrite,
                                                      fun_dir=container_path_to_functions,
                                                      machine_output_f=container_path_to_machine_output,
                                                      testcases_f=container_path_to_testcases,
                                                      strategy=strategy,
                                                      settings_conf=settings_conf,
                                                      bound=bound,
                                                      aliasing_strategy=alias_strategy,
                                                      timeout=timeout)

    if verify_returncode == 0:
        verified_correct, counter_examples_available, counterexample_str = \
            parse_verify_machine_output(container_path_to_machine_output)

        if not verified_correct and counter_examples_available:
            add_counterexample_to_testcases(counterexample_str=counterexample_str,
                                            path_to_testcases=container_path_to_testcases)
    else:
        verified_correct = False
        counter_examples_available = False

    return verified_correct, counter_examples_available
