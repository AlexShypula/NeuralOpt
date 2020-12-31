import os
from time import time
from typing import Dict, Union, List, Tuple
from stoke_helpers import make_tunit_file, test_costfn, verify_and_rewrite_testcase
from utils import STOKE_TRAINING_SET_REGEX
#from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from os.path import join, dirname

NEW_TESTCASE_BEGINNING_INDEX = 2000

class StokePipeline:
    def __init__(self,  n_workers: int, max_cost: int, verification_strategy: str, path_to_volume: str,
                    volume_path_to_data: str, volume_path_to_tmp: str):

        self.n_workers = n_workers
        self.max_cost = max_cost
        self.verification_strategy = verification_strategy
        self.path_to_volume = path_to_volume
        self.volume_path_to_data = volume_path_to_data
        self.volume_path_to_tmp = volume_path_to_tmp

        self.pool = ThreadPoolExecutor(self.n_workers)

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

        cost, failed_tunit, failed_cost, is_correct = get_stoke_cost(hypothesis_string=hypothesis_string,
                                                            container_abs_path_raw_rewrite=container_abs_path_raw_rewrite, 
                                                            container_abs_path_asbly_rewrite=container_abs_path_asbly_rewrite,
                                                            container_abs_path_to_functions=container_abs_path_to_functions,
                                                            container_abs_path_to_target=container_abs_path_to_target,
                                                            container_abs_path_to_testcases=container_abs_path_to_testcases,
                                                            assembly_name=metadata["name"],
                                                            cost_conf=metadata["cost_conf"],
                                                            max_cost=self.max_cost)


        effective_cost = min(cost, self.max_cost)

        beat_baseline_returncode = 0
        # 0 -> didn't beat baseline on cost, 1 -> beat baseline on cost, but doesn't verify, no new testcases
        # 2 -> beat baseline on cost, but doesn't verify, new testcases added,
        # 3 -> beat baseline on cost and verifies correct
        if effective_cost < metadata.get("low_benchmark", self.max_cost) and not failed_tunit and not failed_cost:

            machine_output_filename = rewrite_id + ".verify"
            container_abs_path_machine_output = join(self.path_to_volume, self.volume_path_to_tmp, machine_output_filename)

            next_index = metadata.get("new_testcase_index", NEW_TESTCASE_BEGINNING_INDEX) #TODO: INITIALIZE IN NeuralOpt/model/loss.py

            is_verified_correct, counter_examples_available = verify_and_rewrite_testcase(
                container_path_to_target = container_abs_path_to_target,
                container_path_to_rewrite = container_abs_path_asbly_rewrite,
                container_path_to_testcases = container_abs_path_to_testcases,
                container_path_to_functions = container_abs_path_to_functions,
                container_path_to_machine_output = container_abs_path_machine_output,
                settings_conf = metadata["cost_conf"],
                new_testcase_idx = next_index,
                strategy = self.verification_strategy,
                live_dangerously = True)

            if is_verified_correct:
                print(f"Beat baseline for {metadata['name']} with cost: {effective_cost}, and verified correct",
                      flush = True)
                beat_baseline_returncode = 3

            elif counter_examples_available:
                print(f"New testcases added for {metadata['name']} at index {next_index}", flush=True)
                # inserts in, because the regular expression is simply a lookahead
                # whitespace is necessary following the number here for the STOKE argument parser
                metadata["cost_conf"]["training_set"] = STOKE_TRAINING_SET_REGEX.sub(
                    str(next_index) + " ", metadata["cost_conf"]["training_set"])
                metadata["new_testcase_index"] = next_index + 1
                beat_baseline_returncode = 2

            else:
                print(f"{metadata['name']} beat the baseline, but did not verify", flush=True)
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
                   max_cost = 9999) -> (float, bool, bool):

    with open(os.open(container_abs_path_raw_rewrite, os.O_CREAT | os.O_WRONLY, 0o777), "w+") as fh: # allows full permissions
        fh.write(hypothesis_string)
    tunit_rc, tunit_stdout = make_tunit_file(in_f=container_abs_path_raw_rewrite,
                                             out_f=container_abs_path_asbly_rewrite,
                                             fun_dir=container_abs_path_to_functions,
                                             live_dangerously=True)

    if tunit_rc == 0:

        cost_rc, cost_stdout, cost, correct = test_costfn(
            target_f=container_abs_path_to_target,
            rewrite_f=container_abs_path_asbly_rewrite,
            testcases_f=container_abs_path_to_testcases,
            fun_dir=container_abs_path_to_functions,
            settings_conf=cost_conf,
            live_dangerously=True)

    tunit_failed = False if tunit_rc == 0 else True
    cost_failed = False if tunit_rc == 0 and cost_rc == 0 else True
    correct = False if cost_failed else (correct == "yes") # correct will be a string, and if 'yes' then correct

    if tunit_rc == 0 and cost_rc == 0:
        return float(cost), tunit_failed, cost_failed, correct
    else:
        return float(max_cost), tunit_failed, cost_failed, correct
