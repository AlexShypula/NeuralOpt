from time import time
from typing import Dict
from stoke_helpers import make_tunit_file, test_costfn, verify_and_rewrite_testcase
from multiprocessing.pool import ThreadPool
from os.path import join, dirname

NEW_TESTCASE_BEGINNING_INDEX = 2000

class stoke_pipeline:
    def __init__(self,  n_workers: int, max_cost: int, verification_strategy: str, path_to_volume: str,
                    volume_path_to_data: str, volume_path_to_tmp: str):

        self.n_workers = n_workers
        self.max_cost = max_cost
        self.verification_strategy = verification_strategy
        self.path_to_volume = path_to_volume
        self.volume_path_to_data = volume_path_to_data
        self.volume_path_to_tmp = volume_path_to_tmp

        self.pool = ThreadPool(self.n_workers)

    def run_parallel_pipeline(jobs: Union[List, Tuple], debug = False):
        if debug:
            return self.map(self.run_pipeline_wrapper, jobs)
        else:
            return self.pool.map(self.run_pipeline_wrapper, jobs, chunksize = self.n_workers)

    def run_pipeline_wrapper(kwargs):
        return run_pipeline(**kwargs)

    def run_pipeline(self, hypothesis_string: str, metadata: Dict):

        rewrite_id = (metadata["name"] + "_" + str(time())).replace(".", "_")

        container_abs_path_raw_rewrite = join(path_to_volume, volume_path_to_tmp, rewrite_id + ".tmp"),
        container_abs_path_asbly_rewrite =  join(path_to_volume, volume_path_to_tmp, rewrite_id + ".s"),
        container_abs_path_to_functions = dirname(join(path_to_volume, volume_path_to_data, data_path_to_target)),
        container_abs_path_to_target = join(path_to_volume, volume_path_to_data, data_path_to_target),
        container_abs_path_to_testcases = join(path_to_volume, volume_path_to_data, data_path_to_testcases)}

        cost, failed_tunit, failed_cost = get_stoke_cost(hypothesis_string=hypothesis_string,
                                                            container_abs_path_raw_rewrite=container_abs_path_raw_rewrite
                                                            container_abs_path_asbly_rewrite=container_abs_path_asbly_rewrite,
                                                            container_abs_path_to_functions=container_abs_path_to_functions,
                                                            container_abs_path_to_target=container_abs_path_to_target,
                                                            assembly_name=metadata["name"],
                                                            cost_conf=metadata["cost_conf"],
                                                            max_cost=self.max_cost)


        effective_cost = min(cost, self.max_cost)

        new_record_returncode = 0
        # 0 -> no new record, 1 -> new record, but doesn't verify, no new testcases
        # 2 -> new record, but doesn't verify, new testcases added, 3 -> verifies correct
        if effective_cost < metadata["rolling_baseline_cost"] and not failed_tunit and not failed_cost:

            machine_output_filename = rewrite_id + ".verify"
            container_abs_path_machine_output = join(path_to_volume, volume_path_to_tmp, machine_output_filename)

            next_index = metadata.get("new_testcase_index", NEW_TESTCASE_BEGINNING_INDEX) #TODO: INITIALIZE IN NeuralOpt/model/loss.py

            is_verified_correct, counter_examples_available = verify_and_rewrite_testcase(
                container_path_to_target = container_abs_path_to_target,
                container_path_to_rewrite = container_path_to_rewrite,
                container_path_to_testcases = container_path_to_testcases,
                container_path_to_functions = container_path_to_functions,
                container_path_to_machine_output = container_abs_path_machine_output,
                settings_conf = metadata["cost_conf"],
                new_testcase_idx = next_index,
                strategy = self.verification_strategy,
                live_dangerously = True)

            if is_verified_correct:
                print(f"New record set for {metadata['name']} with cost: {effective_cost}, and verified correct",
                      flush = True)
                metadata["rolling_baseline_cost"] = cost
                new_record_returncode = 3

            elif counter_examples_available:
                print(f"New testcases added for {metadata['name']} at index {next_index}", flush=True)
                # inserts in, because the regular expression is simply a lookahead
                # whitespace is necessary following the number here for the STOKE argument parser
                metadata["cost_conf"]["training_set"] = STOKE_TRAINING_SET_REGEX.sub(
                    str(next_index) + " ", metadata["cost_conf"]["training_set"])
                metadata["new_testcase_index"] = next_index + 1
                new_record_returncode = 2

            else:
                print(f"{metadata['name']} beat the baseline, but did not verify", flush=True)
                new_record_returncode = 1

            if metadata["name"] not in self.asm_names_to_save:
                os.remove(host_abs_path_machine_output)

        if metadata["name"] not in self.asm_names_to_save:
            os.remove(cost_path_dict["host_abs_path_raw_rewrite"])
            os.remove(cost_path_dict["host_abs_path_asbly_rewrite"])

        return metadata, {"cost": effective_cost,
                             "failed_tunit": failed_tunit,
                             "failed_cost": failed_cost,
                             "hypothesis_string": hypothesis_bpe_string,
                             "new_record_returncode": new_record_returncode} # TODO: NEED TO NORMALIZE COST BACK IN NeuralOpt/model/loss.py


def get_stoke_cost(hypothesis_string: str,
                   container_abs_path_raw_rewrite: str,
                   container_abs_path_asbly_rewrite: str,
                   container_abs_path_to_functions: str,
                   container_abs_path_to_target: str,
                   container_abs_path_to_testcases: str,
                   assembly_name: str,
                   cost_conf,
                   max_cost = 9999) -> (float, bool, bool):

    with open(os.open(host_abs_path_raw_rewrite, os.O_CREAT | os.O_WRONLY, 0o777), "w+") as fh: # allows full permissions
        fh.write(formatted_string)
    tunit_rc, tunit_stdout = make_tunit_file(in_f=container_abs_path_raw_rewrite,
                                             out_f=host_abs_path_asbly_rewrite,
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

    if tunit_rc == 0 and cost_rc == 0:
        return float(cost), tunit_failed, cost_failed
    else:
        return float(max_cost), tunit_failed, cost_failed