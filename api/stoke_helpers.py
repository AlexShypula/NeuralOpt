import subprocess
import json
from logging import Logger
from typing import Dict
from typing import Union
from utils import COST_SEARCH_REGEX, CORRECT_SEARCH_REGEX

TIMEOUT=100

def make_tunit_file(in_f: str, out_f: str, fun_dir: str, live_dangerously: bool = False):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    try:
        with open(out_f, "w") as f:
            tunit_proc = subprocess.run(
                 ['/home/stoke/stoke/bin/stoke', 'debug', 'tunit', '--target', in_f,'--functions', fun_dir, "--prune", live_dangerously_str],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT)

    except subprocess.TimeoutExpired as err:
        return -11747, err

    return tunit_proc.returncode, tunit_proc.stdout


def test_costfn(target_f: str,
                rewrite_f: str,
                testcases_f: str,
                fun_dir: str,
                settings_conf: Dict[str, str],
                live_dangerously = True):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
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
                     live_dangerously_str, "--heap_out"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=TIMEOUT)
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
                 live_dangerously_str],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TIMEOUT)
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
                testcases_f: str,
                fun_dir: str,
                settings_conf: Dict[str, str],
                machine_output_f: str,
                strategy: str = "hold_out",
                live_dangerously = True) -> int:
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    try:
        verify_test = subprocess.run(
            ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
             '--target', target_f,
             '--rewrite', rewrite_f,
             '--machine_output', machine_output_f,
             '--strategy', strategy,
             '--testcases', testcases_f,
             '--functions', fun_dir,
             "--prune", live_dangerously_str,
             "--def_in", settings_conf["def_in"],
             "--live_out", settings_conf["live_out"],
             "--distance", settings_conf["distance"],
             "--misalign_penalty", str(settings_conf["misalign_penalty"]),
             "--sig_penalty", settings_conf["sig_penalty"],
             "--cost", settings_conf["costfn"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300)
        return verify_test.returncode
    except subprocess.TimeoutExpired as err:
        print(f"verify timed out with error {err}")
        return -1


def parse_verify_machine_output(machine_output_f: str) -> (bool, bool, Union[None, str]):
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)

    verified_correct = machine_output_dict["verified"]
    counter_examples_available = machine_output_dict["counter_examples_available"]
    if not verified_correct:
        if counter_examples_available:
            counterexample_str = machine_output_dict["counterexample"]
        else:
            counterexample_str = None
    else:
        counterexample_str = None
    return verified_correct, counter_examples_available, counterexample_str


def add_counterexample_to_testcases(counterexample_str: str, path_to_testcases: str, new_testcase_idx: int):
    with open(path_to_testcases, "a") as fh:
        fh.write(f"\n\n\nTestcase {str(new_testcase_idx)}\n\n:")
        fh.write(counterexample_str)


def verify_and_rewrite_testcase(container_path_to_target: str,
                                container_path_to_rewrite: str,
                                container_path_to_testcases: str,
                                container_path_to_functions: str,
                                container_path_to_machine_output: str,
                                settings_conf: Dict[str, str],
                                new_testcase_idx: int,
                                strategy: str = "hold_out",
                                live_dangerously = True):


    verify_returncode =  verify_rewrite(target_f=container_path_to_target,
                                        rewrite_f=container_path_to_rewrite,
                                        testcases_f=container_path_to_testcases,
                                        fun_dir=container_path_to_functions,
                                        settings_conf=settings_conf,
                                        machine_output_f=container_path_to_machine_output,
                                        strategy=strategy,
                                        live_dangerously=live_dangerously)
    if verify_returncode == 0:
        is_verified_correct, counter_examples_available, counterexample_str = parse_verify_machine_output(container_path_to_machine_output)

        if is_verified_correct and counter_examples_available:
            add_counterexample_to_testcases(counterexample_str=counterexample_str,
                                            path_to_testcases=container_path_to_testcases,
                                            new_testcase_idx=new_testcase_idx)
    else:
        is_verified_correct = counter_examples_available = False


    return is_verified_correct, counter_examples_available
