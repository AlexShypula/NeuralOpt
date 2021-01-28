import os
from time import time
from typing import Dict, Union, List, Tuple
from stoke_helpers import make_tunit_file, test_costfn, verify_and_rewrite_testcase, verify_rewrite, \
			parse_verify_machine_output
from utils import STOKE_TRAINING_SET_REGEX, mkdir
#from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from os.path import join, dirname
import warnings


NEW_TESTCASE_BEGINNING_INDEX = 2000

class StokePipeline:
    def __init__(self,  n_workers: int, max_cost: int, verification_strategy: str, path_to_volume: str,
                    volume_path_to_data: str, volume_path_to_tmp: str, alias_strategy: str = None,
                    bound: int = None, cost_timeout: int = 100, verification_timeout: int = 300):

        self.n_workers = n_workers
        self.max_cost = max_cost
        assert verification_strategy in ("hold_out", "bounded", "ddec"), "unsupported verification_strategy"
        if verification_strategy == "ddec":
            warnings.warn("Warning: data driven equivalence checking in STOKE is experimental and reported to be buggy")
        self.verification_strategy = verification_strategy
        self.path_to_volume = path_to_volume
        self.volume_path_to_data = volume_path_to_data
        self.volume_path_to_tmp = volume_path_to_tmp
        self.cost_timeout = cost_timeout
        self.verification_timeout = verification_timeout

        if self.verification_strategy == "bounded":
            assert type(bound) == int and bound > 0, "if using a formal validator, you'll need to specify the bound"
            assert alias_strategy in ("basic","string", "string_antialias","flat"), "if using a formal validator you"\
                                                        "must specify the aliasing strategy. Basic is recommended"
        self.bound = bound
        self.alias_strategy = alias_strategy

        self.pool = ThreadPoolExecutor(self.n_workers)

        mkdir(join(self.path_to_volume, self.volume_path_to_tmp, "verified_bound4"))

    def run_parallel_eval(self, jobs: Union[List, Tuple], debug=False):
        if debug:
            return map(self.run_eval_wrapper, jobs)
        else:
            return self.pool.map(self.run_eval_wrapper, jobs, chunksize = self.n_workers)


    def run_eval_wrapper(self, kwargs):
        return self.run_eval(**kwargs)


    def run_eval(self, hypothesis_string: str, metadata: Dict):

        rewrite_id = (metadata["name"] + "_" + str(time())).replace(".", "_")
        data_path_to_target = metadata["base_asbly_path"]
        data_path_to_testcases = metadata["testcase_path"]

        container_abs_path_raw_rewrite = join(self.path_to_volume, self.volume_path_to_tmp, rewrite_id + ".tmp")
        container_abs_path_asbly_rewrite =  join(self.path_to_volume, self.volume_path_to_tmp, rewrite_id + ".s")
        container_abs_path_to_functions = dirname(join(self.path_to_volume, self.volume_path_to_data, data_path_to_target))
        container_abs_path_to_target = join(self.path_to_volume, self.volume_path_to_data, data_path_to_target)
        container_abs_path_to_testcases = join(self.path_to_volume, self.volume_path_to_data, data_path_to_testcases)

        metadata["cost_conf"]["training_set"] = "{ 0 ... 9999 }"
        metadata["cost_conf"]["heap_out"] = True
        assert os.path.exists(container_abs_path_to_functions), "paths to functions doesn't exist"
        assert os.path.exists(container_abs_path_to_target), "paths to target"
        assert os.path.exists(container_abs_path_to_testcases), "paths to testcases"

        cost, failed_tunit, failed_cost, is_correct = get_stoke_cost(hypothesis_string=hypothesis_string,
                                                            container_abs_path_raw_rewrite=container_abs_path_raw_rewrite,
                                                            container_abs_path_asbly_rewrite=container_abs_path_asbly_rewrite,
                                                            container_abs_path_to_functions=container_abs_path_to_functions,
                                                            container_abs_path_to_target=container_abs_path_to_target,
                                                            container_abs_path_to_testcases=container_abs_path_to_testcases,
                                                            assembly_name=metadata["name"],
                                                            cost_conf=metadata["cost_conf"],
                                                            max_cost=1e9)

        if is_correct and self.verification_strategy == "bounded":
            machine_output_filename = rewrite_id + ".verify"
            container_abs_path_machine_output = join(self.path_to_volume, self.volume_path_to_tmp, machine_output_filename)
            verify_returncode, verify_stdout = verify_rewrite(target_f=container_abs_path_to_target,
                                                      rewrite_f=container_abs_path_asbly_rewrite,
                                                      fun_dir=container_abs_path_to_functions,
                                                      machine_output_f=container_abs_path_machine_output,
                                                      testcases_f=container_abs_path_to_testcases,
                                                      strategy=self.verification_strategy,
                                                      settings_conf=metadata["cost_conf"],
                                                      bound=self.bound,
                                                      aliasing_strategy=self.alias_strategy,
                                                      timeout=self.verification_timeout)

            if verify_returncode == 0:
                is_correct, counter_examples_available, counterexample_str = \
                                    parse_verify_machine_output(container_abs_path_machine_output)
            else:
                is_correct = False
            os.remove(container_abs_path_machine_output)

        print("cost is {}".format(cost))
        print("failed tunit is {}".format(failed_tunit), flush = True)

        return {"metadata": metadata, "stats": {"cost": cost,
                                         "correct": is_correct,
                                         "failed_tunit": failed_tunit,
                                         "failed_cost": failed_cost,
                                         "hypothesis_string": hypothesis_string}}


    def run_parallel_pipeline(self, jobs: Union[List, Tuple], debug = False):
        if debug:
            return map(self.run_pipeline_wrapper, jobs)
        else:
            return self.pool.map(self.run_pipeline_wrapper, jobs, chunksize = self.n_workers)

    def run_pipeline_wrapper(self, kwargs):
        return self.run_pipeline(**kwargs)

    def run_pipeline(self, hypothesis_string: str, metadata: Dict):

        rewrite_id = (metadata["name"] + "_" + str(time())).replace(".", "_")
        data_path_to_target = metadata["base_asbly_path"]
        data_path_to_testcases = metadata["testcase_path"]

        container_abs_path_raw_rewrite = join(self.path_to_volume, self.volume_path_to_tmp, rewrite_id + ".tmp")
        container_abs_path_asbly_rewrite =  join(self.path_to_volume, self.volume_path_to_tmp, rewrite_id + ".s")
        container_abs_path_to_functions = dirname(join(self.path_to_volume, self.volume_path_to_data, data_path_to_target))
        container_abs_path_to_target = join(self.path_to_volume, self.volume_path_to_data, data_path_to_target)
        container_abs_path_to_testcases = join(self.path_to_volume, self.volume_path_to_data, data_path_to_testcases)
        # patch here to use all testcases
        metadata["cost_conf"]["training_set"] = "{ 0 ... 9999 }"

        cost, failed_tunit, failed_cost, is_correct = get_stoke_cost(hypothesis_string=hypothesis_string,
                                                            container_abs_path_raw_rewrite=container_abs_path_raw_rewrite, 
                                                            container_abs_path_asbly_rewrite=container_abs_path_asbly_rewrite,
                                                            container_abs_path_to_functions=container_abs_path_to_functions,
                                                            container_abs_path_to_target=container_abs_path_to_target,
                                                            container_abs_path_to_testcases=container_abs_path_to_testcases,
                                                            assembly_name=metadata["name"],
                                                            cost_conf=metadata["cost_conf"],
                                                            max_cost=self.max_cost,
                                                            timeout=self.cost_timeout)


        effective_cost = min(cost, self.max_cost)

        beat_baseline_returncode = 0
        # 0 -> didn't beat baseline on cost, 1 -> beat baseline on cost, but doesn't verify, no new testcases
        # 2 -> beat baseline on cost, but doesn't verify, new testcases added,
        # 3 -> beat baseline on cost and verifies correct
        if effective_cost < metadata.get("low_benchmark", self.max_cost) and not failed_tunit and not failed_cost:

            machine_output_filename = rewrite_id + ".verify"
            container_abs_path_machine_output = join(self.path_to_volume, self.volume_path_to_tmp, machine_output_filename)


            is_verified_correct, counter_examples_available, verify_stdout = verify_and_rewrite_testcase(
                container_path_to_target = container_abs_path_to_target,
                container_path_to_rewrite = container_abs_path_asbly_rewrite,
                container_path_to_testcases = container_abs_path_to_testcases,
                container_path_to_functions = container_abs_path_to_functions,
                container_path_to_machine_output = container_abs_path_machine_output,
                settings_conf = metadata["cost_conf"],
                strategy = self.verification_strategy,
                alias_strategy = self.alias_strategy,
                bound = self.bound,
                timeout = self.verification_timeout
                )

            if is_verified_correct:
                print(f"Beat baseline for {metadata['name']} with cost: {effective_cost}, and verified correct",
                      flush = True)
                beat_baseline_returncode = 3
                with open(join(self.path_to_volume, self.volume_path_to_tmp, "verified_bound4", \
                           os.path.splitext(metadata["name"])[0]+".verified"), "w+") as fh:
                    fh.write("Program: {}\n".format(metadata["name"]))
                    fh.write("Live out: {}\n".format(metadata["cost_conf"]["live_out"]))
                    fh.write("Heap out: {}\n".format(metadata["cost_conf"]["heap_out"]))
                    fh.write("verify output is :\n\n{}".format(verify_stdout))

            elif counter_examples_available:
                print(f"New testcases added for {metadata['name']} ", flush=True)
                # inserts in, because the regular expression is simply a lookahead
                # whitespace is necessary following the number here for the STOKE argument parser
#                 metadata["cost_conf"]["training_set"] = STOKE_TRAINING_SET_REGEX.sub(
#                     str(next_index) + " ", metadata["cost_conf"]["training_set"])
#                 metadata["new_testcase_index"] = next_index + 1
                is_correct = False
                beat_baseline_returncode = 2

            else:
                print(f"{metadata['name']} beat the baseline, but did not verify", flush=True)
                is_correct = False
                beat_baseline_returncode = 1

            if not metadata.get("save_intermediate_flag"):
                os.remove(container_abs_path_machine_output)

        if not metadata.get("save_intermediate_flag"):
            os.remove(container_abs_path_raw_rewrite)
            os.remove(container_abs_path_asbly_rewrite)

        return {"metadata": metadata, "stats": {"cost": effective_cost,
                                                 "failed_tunit": failed_tunit,
                                                 "failed_cost": failed_cost,
						                         "correct": is_correct,
                                                 "hypothesis_string": hypothesis_string,
                                                 "beat_baseline_returncode": beat_baseline_returncode}}


def get_stoke_cost(hypothesis_string: str,
                   container_abs_path_raw_rewrite: str,
                   container_abs_path_asbly_rewrite: str,
                   container_abs_path_to_functions: str,
                   container_abs_path_to_target: str,
                   container_abs_path_to_testcases: str,
                   assembly_name: str,
                   cost_conf: Dict,
                   max_cost: int = 9999,
                   timeout: int = 100) -> (float, bool, bool):

    with open(os.open(container_abs_path_raw_rewrite, os.O_CREAT | os.O_WRONLY, 0o777), "w+") as fh: # allows full permissions
        fh.write(hypothesis_string)
    tunit_rc, tunit_stdout = make_tunit_file(in_f=container_abs_path_raw_rewrite,
                                             out_f=container_abs_path_asbly_rewrite,
                                             fun_dir=container_abs_path_to_functions,
                                             timeout=timeout)

    if tunit_rc == 0:

        cost_rc, cost_stdout, cost, correct = test_costfn(
            target_f=container_abs_path_to_target,
            rewrite_f=container_abs_path_asbly_rewrite,
            testcases_f=container_abs_path_to_testcases,
            fun_dir=container_abs_path_to_functions,
            settings_conf=cost_conf,
            timeout=timeout)

    tunit_failed = False if tunit_rc == 0 else True
    cost_failed = False if tunit_rc == 0 and cost_rc == 0 else True
    correct = False if cost_failed else (correct == "yes") # correct will be a string, and if 'yes' then correct

    if tunit_rc == 0 and cost_rc == 0:
        return float(cost), tunit_failed, cost_failed, correct
    else:
        return float(max_cost), tunit_failed, cost_failed, correct
