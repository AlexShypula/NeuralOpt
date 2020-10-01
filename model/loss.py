# coding: utf-8
"""
Module to implement training loss
"""

import torch
import torch.nn.functional as F
from torch.distributions.utils import clamp_probs
import numpy as np
import re
import pickle
from torch import nn, Tensor
from torch.autograd import Variable
from time import time
from os.path import join, basename, dirname
from helpers import mkdir, hash_file, make_tunit_file, test_costfn, bpe2formatted, PriorityQueue, \
    verify_and_rewrite_testcase, make_verify_rewrite_paths, make_cost_paths, function_path_to_unique_name, \
    annotate_eval_string
from collections import deque
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Dict
import os
from copy import copy
from req import StokeRequest
import warnings
import matplotlib.pyplot as plt

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

STOKE_TRAINING_SET_REGEX = re.compile("(?=})") # when you do sub, you need to include an extra space after the integer

rc2axis = {-2: "not trained on yet",
           -1: "failed",
           0: "incorrect",
           1: "worse than -O0",
           2: "matched -O0",
           3: "between -O0 and -Og",
           4: "matched -Og",
           5: "better than -Og"}


def log_probs_and_entropy(logits, labels, loss_mask, clamp = False):
    dims = logits.size()
    all_log_probs = F.log_softmax(logits, dim = -1)
    probs = torch.exp(all_log_probs)
    if clamp:
        probs = clamp_probs(probs)
    entropy = -1 * torch.sum(all_log_probs * probs * loss_mask.unsqueeze(2)) # broadcast loss mash whcih is B X T -> B X T X 1, while probs are B x T x out_dim
    action_log_probs = -1 * F.nll_loss(all_log_probs.reshape(-1, dims[2]), labels.reshape(-1), reduction="none")
    action_log_probs = action_log_probs.reshape(dims[0], dims[1])
    action_log_probs = action_log_probs * loss_mask
    return action_log_probs, entropy


class StokeCostManager:
    def __init__(self, hash2metadata, host_path_to_volume, container_path_to_volume,
                 volume_path_to_data, volume_path_to_tmp, tb_writer, n_best_seq_dir, trailing_stats_out_path: str,
                 baseline_cost_key: str, asm_names_to_save: List[str] = [], verifiction_strategy: str = "hold_out",
                 new_testcase_beginning_index: int = 2000, max_len = 256, max_score = 9999,
                 n_workers=8, keep_n_best_seqs=5, n_testcases = 32, container_port = "6000",
                 trailing_stats_in_path: str = None):
        self.hash2metadata = hash2metadata
        # self.container_name = container_name
        self.container_port = container_port
        self.requester = StokeRequest(base_url = "http://127.0.0.1", port = self.container_port)
        self.host_path_to_volume = host_path_to_volume
        self.container_path_to_volume = container_path_to_volume
        self.volume_path_to_data = volume_path_to_data
        self.volume_path_to_tmp = volume_path_to_tmp
        mkdir(join(self.host_path_to_volume, self.volume_path_to_tmp))
        self.tb_writer = tb_writer
        self.n_best_seq_dir = n_best_seq_dir
        self.trailing_stats_out_path = trailing_stats_out_path
        mkdir(self.n_best_seq_dir)
        self.baseline_cost_key = baseline_cost_key
        self.asm_names_to_save = asm_names_to_save
        self.verification_strategy = verifiction_strategy
        self.new_testcase_beginning_index = new_testcase_beginning_index
        self.n_workers = n_workers
        self.trailing_stats_dict = dict()
        self.val_step = 0
        self.priority_queue_length = keep_n_best_seqs
        self.n_testcases = n_testcases
        self.trailing_stats_in_path = trailing_stats_in_path
        self.max_score = max_score
        self.beat_baselines_hash_set = set()
        self.no_beat_baselines = 0

        if self.trailing_stats_in_path:
            with open(self.trailing_stats_in_path, "rb") as f:
                self.trailing_stats_dict = pickle.load(f)

        for h in hash2metadata.keys():
            self.hash2metadata[h]["name"] = function_path_to_unique_name(self.hash2metadata[h]["base_asbly_path"]) if \
                not self.hash2metadata[h].get("name") else self.hash2metadata[h].get("name")
            self.hash2metadata[h]["name"] = basename(self.hash2metadata[h]["base_asbly_path"])
            self.hash2metadata[h]["cost_conf"]["training_set"] = f"{{ 0 ... {n_testcases-1} }}"
            self.hash2metadata[h]["rolling_baseline_cost"] = copy(self.hash2metadata[h][self.baseline_cost_key])
            self.hash2metadata[h]['reference_score'] = copy(self.hash2metadata[h][self.baseline_cost_key])
            # TODO undo the hard-coding here
            self.hash2metadata[h]['low_benchmark'] = min(self.hash2metadata[h]["O0_cost"], self.hash2metadata[h]["Og_cost"])
            self.hash2metadata[h]['high_benchmark'] = max(self.hash2metadata[h]["O0_cost"], self.hash2metadata[h]["Og_cost"])
            self.hash2metadata[h]['best_cost_so_far'] = 1e9
            self.hash2metadata[h]['best_seq_returncode'] = -2


            if not self.trailing_stats_in_path:
                self.trailing_stats_dict[h] = {"costs": deque(maxlen = max_len),
                                                "failed_tunit": deque(maxlen = max_len),
                                                "failed_cost": deque(maxlen = max_len),
                                                "normalized_advantage": deque(maxlen = max_len),
                                                "n_steps": 0,
                                                "best_sequence_priority_queue": PriorityQueue(maxlen=self.priority_queue_length)}


    def get_mean_stdv_cost(self, h: str):

        cost_std = 1 if len(self.trailing_stats_dict[h]["costs"]) < 2 else np.std(
            self.trailing_stats_dict[h]["costs"])
        cost_mean = 0 if len(self.trailing_stats_dict[h]["costs"]) < 2 else np.mean(
            self.trailing_stats_dict[h]["costs"])

        return cost_mean, cost_std


    def parallel_get_rl_cost(self, bpe_strings: List[Tuple[str, str]]):
        jobs = {}
        for (source_bpe_str, hypothesis_bpe_str) in bpe_strings:
            h = hash_file(source_bpe_str.strip())
            if h in jobs:
                print(f"duplicate for {self.hash2metadata[h]['name']}")
            #     breakpoint()
            # assert h not in jobs, '''batches must only have only one sample for each observation
            #                             in order to include multiple samples, this needs to be done in an outerloop
            #                             with gradient accumulation'''
            metadata = self.hash2metadata[h]
            formatted_hypothesis, _ = bpe2formatted(assembly_string = hypothesis_bpe_str, function_name = metadata["name"],
                                                 remove_header = True, remove_footer = True)
            jobs[h] = {"hypothesis_string": formatted_hypothesis, "metadata": metadata}
        results = self.requester.get(jobs)
        hashes_advantages_stats = []
        for h, result_dict in results.items():
            self.hash2metadata[h] = result_dict["metadata"]
            stats = result_dict["stats"]
            # cost_std = 1 if len(self.trailing_stats_dict[h]["costs"]) == 0 else np.std(
            #     self.trailing_stats_dict[h]["costs"])
            cost_std = 1
            cost_mean = 0 if len(self.trailing_stats_dict[h]["costs"]) == 0 else np.mean(
                self.trailing_stats_dict[h]["costs"])
            normalized_advantage = (stats["cost"] - cost_mean) / (cost_std if cost_std != 0 else 1)
            stats["normalized_advantage"] = normalized_advantage
            stats["hypothesis_string"] = jobs[h]["hypothesis_string"]
            hashes_advantages_stats.append((h, normalized_advantage, stats))
        return hashes_advantages_stats

    def eval_beams(self, source_bpe_str: str, hyp_bpe_beams: str):
        jobs = {}
        h = hash_file(source_bpe_str.strip())
        metadata = self.hash2metadata[h]
        formatted_src, _ = bpe2formatted(assembly_string = source_bpe_str, function_name = metadata["name"],
                                      remove_header = True, remove_footer = True)
        for i, hyp in enumerate(hyp_bpe_beams):
            formatted_hyp, _ = bpe2formatted(assembly_string = hyp, function_name = metadata["name"],
                                          remove_header = True, remove_footer = True)
            jobs[i] = {"hypothesis_string": formatted_hyp, "metadata": metadata}
        results = self.requester.eval(jobs)
        ##TODO YOU NEED TO SPECIFY HOW TO GET THE O0 and Og benchmarks here !
        rc = -1
        unopt_cost = self.hash2metadata[h]["O0_cost"]
        opt_cost = self.hash2metadata[h]["Og_cost"]
        high_benchmark = max(unopt_cost, opt_cost)
        low_benchmark = min(unopt_cost, opt_cost)
        best_cost = 1e9
        best_result = {"hypothesis_string": jobs[0]["hypothesis_string"], "stats": results["0"]["stats"]}
        for i, result_dict in results.items():
            i = int(i) # quirks of conversion int -> string
            cost = result_dict["stats"]["cost"]
            correct = result_dict["stats"]["correct"]
            failed = result_dict["stats"]["failed_cost"]
            if cost < best_cost:
                if rc < 0 and failed:
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                if rc < 0 and not correct and not failed:
                    rc = 0
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                elif rc < 1 and cost > high_benchmark and correct:
                    rc = 1
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                elif rc < 2 and cost == high_benchmark and correct:
                    rc = 2
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                elif rc < 3 and cost < high_benchmark and cost > low_benchmark and correct:
                    assert cost <= high_benchmark
                    rc = 3
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                elif rc < 4 and cost == low_benchmark and correct:
                    assert cost <= high_benchmark
                    rc = 4
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                elif rc < 5 and cost < low_benchmark and correct:
                    assert cost < low_benchmark and cost < high_benchmark
                    rc = 5
                    best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                    best_cost = cost
                else:
                    if cost < best_result["stats"]["cost"]:
                        best_result = {"hypothesis_string": jobs[i]["hypothesis_string"], "stats": result_dict["stats"]}
                        best_cost = cost
                    else: 
                        raise ValueError

        comparison_string = annotate_eval_string(reference_string = formatted_src,
                                                                hypothesis_string = best_result["hypothesis_string"],
                                                                function_name=metadata["name"],
                                                                best_cost = best_result["stats"]["cost"],
                                                                unopt_cost = unopt_cost,
                                                                opt_cost = opt_cost,
                                                                correctness_flag = best_result["stats"]["correct"],
                                                                return_code = rc)

        return rc, best_result["hypothesis_string"], best_result["stats"], comparison_string, metadata

    def batch_eval_greedy(self, bpe_strings: List[Tuple[str, str]]):
        jobs = {}
        index2src = {}
        for i, (source_bpe_str, hypothesis_bpe_str) in enumerate(bpe_strings):
            index2src[i] = source_bpe_str
            h = hash_file(source_bpe_str.strip())
            if h in jobs:
                print(f"duplicate for {self.hash2metadata[h]['name']}")

            metadata = self.hash2metadata[h]

            formatted_hypothesis, _ = bpe2formatted(assembly_string = hypothesis_bpe_str, function_name = metadata["name"],
                                                 remove_header = True, remove_footer = True)

            jobs[i] = {"hypothesis_string": formatted_hypothesis, "metadata": metadata}
        results = self.requester.eval(jobs)

        result_tuples = []
        for i, result_dict in results.items():
            metadata = result_dict["metadata"]
            source_bpe_str = index2src[i]

            unopt_cost = metadata["O0_cost"]
            opt_cost = metadata["Og_cost"]
            high_benchmark = max(unopt_cost, opt_cost)
            low_benchmark = min(unopt_cost, opt_cost)

            cost = result_dict["stats"]["cost"]
            correct = result_dict["stats"]["correct"]
            failed = result_dict["stats"]["failed_cost"]

            if failed:
                rc = -1
            elif not correct:
                rc = 0
            elif cost > high_benchmark:
                rc = 1
            elif rc < 2 and cost == high_benchmark:
                rc = 2
            elif rc < 3 and cost > low_benchmark:
                assert cost <= high_benchmark
                rc = 3
            elif rc < 4 and cost == low_benchmark:
                assert cost <= high_benchmark
                rc = 4
            elif rc < 5 and cost < low_benchmark:
                assert cost < low_benchmark and cost < high_benchmark
                rc = 5
            else:
                raise Exception

            formatted_source, _ = bpe2formatted(assembly_string=source_bpe_str, function_name=metadata["name"],
                                                remove_header=True, remove_footer=True)
            formatted_hypothesis = jobs[i]["hypothesis_string"]

            comparison_string = annotate_eval_string(reference_string = formatted_source,
                                                        hypothesis_string = formatted_hypothesis,
                                                        function_name=metadata["name"],
                                                        best_cost = cost,
                                                        unopt_cost = unopt_cost,
                                                        opt_cost = opt_cost,
                                                        correctness_flag = correct, 
                                                        return_code = rc)


            # result_tuples.append((rc, {"hypothesis_string": jobs[i]["hypothesis_string"],
            #                            "stats": result_dict["stats"],
            #                            "comparison_string": comparison_string,
            #                            "metadata": metadata}))

            result_tuples.append(rc, formatted_hypothesis, result_dict["stats"], comparison_string, metadata)

        return result_tuples

    def _get_updated_rewrite_returncode(self, rewrite_cost, rewrite_failed, rewrite_correct, metadata):
        old_returncode = metadata["best_seq_returncode"]
        best_cost_so_far = metadata["best_cost_so_far"]
        new_returncode = old_returncode
        if old_returncode < -1:
            new_returncode = -1 # -2 -> not trained on yet
        if old_returncode < 0 and not rewrite_failed:
            new_returncode = 0 # -1 -> failed, so this re-write did not fail
        if old_returncode < 1 and rewrite_correct:
            new_returncode = 1 # 1 is the baseline meaning, this is worse than O0, we try to see if we
                                    # need to modify this next
        if rewrite_cost < best_cost_so_far and rewrite_correct and not rewrite_failed:
            best_cost_so_far = rewrite_cost
            low_benchmark = metadata["low_benchmark"]
            high_benchmark = metadata["high_benchmark"]

            # use if instead of the elif paradigm because it is possible O0 == Og
            if old_returncode < 1 and rewrite_cost > high_benchmark:
                new_returncode = 1
            if old_returncode < 2 and rewrite_cost == high_benchmark:
                new_returncode = 2
            if old_returncode < 3 and rewrite_cost < high_benchmark and rewrite_cost > low_benchmark:
                assert rewrite_cost <= high_benchmark
                new_returncode = 3
            if old_returncode < 4 and rewrite_cost == low_benchmark:
                assert rewrite_cost <= high_benchmark
                new_returncode = 4
            elif old_returncode < 5 and rewrite_cost < low_benchmark:
                assert rewrite_cost < low_benchmark and rewrite_cost < high_benchmark
                new_returncode = 5

        return new_returncode, best_cost_so_far

    def _calculate_best_seq_statistics(self):
        rc_stats_dict = {}

        for i in range(-2, 6):
            rc_stats_dict[i] = {"counts": 0, "costs": []}
        #breakpoint()
        for metadata in self.hash2metadata.values():
            best_seq_returncode = metadata["best_seq_returncode"]
            best_cost_so_far = metadata["best_cost_so_far"]
            rc_stats_dict[best_seq_returncode]["counts"]+=1
            rc_stats_dict[best_seq_returncode]["costs"].append(best_cost_so_far)
        #breakpoint()
        percentage_dict = {}
        cost_dict = {}

        for rc, stats in rc_stats_dict.items():
            mean_cost = np.mean(stats["costs"]) if len(stats["costs"]) > 0 else -1
            n_seqs = len(self.hash2metadata) - rc_stats_dict[-2]["counts"]
            pct_of_trained = stats["counts"] / n_seqs
            if rc != -2:
                title = rc2axis[rc]
                percentage_dict[title] = pct_of_trained * 100
                cost_dict[title] = mean_cost

        return percentage_dict, cost_dict

    def _plot_best_seq_stats(self, percentage_dict, cost_dict):
        output_path = os.path.dirname(self.trailing_stats_out_path) # by default model dir

        # PERCENTAGE DICT
        plt.rcParams["font.family"] = "sans-serif"
        plt.rc('axes', axisbelow=True)

        plt.grid(color='gray', linestyle='dashed')
        max_val = max(percentage_dict.values())
        annotation_dict = {k: (f'{v:.2f}%' if v != -1 else "NA") for k, v in percentage_dict.items()}
        textstr = "Percentages per Bucket:" + "\n\n" + '\n'.join([f"{k} = {v}" for k, v in annotation_dict.items()])
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(7, max_val, textstr, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.bar(percentage_dict.keys(), percentage_dict.values(), color="darkgreen")
        plt.title("Distribution of Result Types During Training")
        plt.ylabel("Percentage")
        plt.xticks(rotation=75)
        plt.savefig(join(output_path, "percentage.png"), dpi=300, pad_inches=2, bbox_inches="tight")
        plt.clf()

        # COST DICT
        max_val = max(cost_dict.values())

        plt.rc('axes', axisbelow=True)
        plt.rcParams["font.family"] = "sans-serif"
        plt.grid(color='gray', linestyle='dashed')

        plt.bar(cost_dict.keys(), cost_dict.values(), color="darkgreen")
        plt.title("Average Stoke Cost by Performance Bucket During Training")
        plt.ylabel("Stoke Cost")
        plt.yscale("log")
        plt.xticks(rotation=75)

        annotation_dict = {k: (f'{v:.2f}' if v != -1 else "NA") for k, v in cost_dict.items()}
        textstr = "Costs per Bucket:" + "\n\n" + '\n'.join([f"{k} = {v}" for k, v in annotation_dict.items()])
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.text(7, max_val, textstr, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.savefig(join(output_path, "cost.png"), dpi=300, pad_inches=2, bbox_inches="tight")
        plt.clf()

    def save_best_seq_stats(self):
        #breakpoint()
        percentage_dict, cost_dict = self._calculate_best_seq_statistics()
        #breakpoint()
        self._plot_best_seq_stats(percentage_dict=percentage_dict, cost_dict=cost_dict)

    def update_buffers(self, hash_stats_list: Tuple[str, Dict]):

        batch_cost = 0
        pct_failures = 0
        for h, stats in hash_stats_list:
            normalized_advantage = stats["normalized_advantage"]
            effective_cost = stats["cost"]
            failed_tunit = stats["failed_tunit"]
            failed_cost = stats["failed_cost"]
            hypothesis_string = stats["hypothesis_string"]
            new_record_returncode = stats["new_record_returncode"]
            correct = stats["correct"]
            #print("failed cost is {}, correct is {}, and effective cost is {}".format(failed_cost, correct, effective_cost))
            #breakpoint()
            best_seq_returncode, best_cost_so_far = self._get_updated_rewrite_returncode(
                                                                            rewrite_cost = effective_cost,
                                                                            rewrite_failed = failed_cost,
                                                                            rewrite_correct = correct,
                                                                            metadata = self.hash2metadata[h]
                                                                                            )
            self.hash2metadata[h]["best_seq_returncode"] = best_seq_returncode
            self.hash2metadata[h]["best_cost_so_far"] = best_cost_so_far


            if new_record_returncode == 3:
                print(f"for {self.hash2metadata[h]['name']} the baseline was beat and verified")
                print(f"the cost was {stats['cost']} whereas the reference was {self.hash2metadata[h]['reference_score']}")
                print(f"the rolling baseline is now {self.hash2metadata[h]['rolling_baseline_cost']}", flush = True)
                beat_baseline_str = f"Beat Baseline and verified, " \
                                    f"where reference is {self.hash2metadata[h]['reference_score']}\n"
                if h not in self.beat_baselines_hash_set:
                    self.beat_baselines_hash_set.add(h)
                    self.no_beat_baselines+=1
            elif new_record_returncode in (1,2):
                print(f"for {self.hash2metadata[h]['name']} the baseline was beat, but didn't verify")
                print(f"the cost was {stats['cost']} whereas the reference was {self.hash2metadata[h]['reference_score']}")
                print(f"the rolling baseline is still {self.hash2metadata[h]['rolling_baseline_cost']}", flush = True)
                beat_baseline_str = f"Beat Baseline but didn't verify, " \
                                    f"where reference is {self.hash2metadata[h]['reference_score']}\n"
            else:
                beat_baseline_str = ""


            # update the buffers
            self.trailing_stats_dict[h]["normalized_advantage"].append(normalized_advantage)
            self.trailing_stats_dict[h]["costs"].append(effective_cost)
            self.trailing_stats_dict[h]["failed_tunit"].append(failed_tunit)
            self.trailing_stats_dict[h]["failed_cost"].append(failed_cost)
            self.trailing_stats_dict[h]["best_sequence_priority_queue"].append(effective_cost,
                                                                               beat_baseline_str + hypothesis_string)
            # update


            batch_cost += effective_cost
            pct_failures += failed_cost
        batch_cost /= len(hash_stats_list)
        pct_failures /= len(hash_stats_list)
        return batch_cost, pct_failures

    def log_buffer_stats(self, hash_list: List[str]):

        if hash_list == None:
            warnings.warn("In logging buffer statistics, no hash keys were given,"
                          "the entire dataset is now being logged each update")
            hash_list = self.trailing_stats_dict.keys()

        for h in hash_list:

            if len(self.trailing_stats_dict[h]["costs"]) > 0:

                name = self.hash2metadata[h]["name"]
                # re-calculate stats for logger
                mean_normalized_advantage = np.mean(self.trailing_stats_dict[h]["normalized_advantage"])
                trailing_cost_mean = np.mean(self.trailing_stats_dict[h]["costs"])
                trailing_cost_std = np.std(self.trailing_stats_dict[h]["costs"])
                trailing_failed_tunit = np.mean(self.trailing_stats_dict[h]["failed_tunit"])
                trailing_failed_cost = np.mean(self.trailing_stats_dict[h]["failed_cost"])
                step_no = self.trailing_stats_dict[h]["n_steps"]

                self.tb_writer.add_scalar(f"{name}/normalized_advantage", mean_normalized_advantage, step_no)
                self.tb_writer.add_scalar(f"{name}/trailing_cost", trailing_cost_mean, step_no)
                self.tb_writer.add_scalar(f"{name}/trailing_std", trailing_cost_std, step_no)
                self.tb_writer.add_scalar(f"{name}/trailing_failed_tunit", trailing_failed_tunit, step_no)
                self.tb_writer.add_scalar(f"{name}/trailing_failed_cost", trailing_failed_cost, step_no)

                self.trailing_stats_dict[h]["n_steps"] += 1

    def _write_n_best(self, name: str , priority_queue: PriorityQueue):
        with open(join(self.n_best_seq_dir, name + "_best.txt"), "w") as fh:
            fh.write(f"Last val step updated: {self.val_step}\n")
            for i, (neg_cost, sequence) in enumerate(sorted(priority_queue.queue, reverse=True)):
                fh.write(f"\n\nRank {i} best sequence for problem {name} has cost: {-neg_cost}\n{'-'*40}\n\n")
                fh.write(f"{sequence}\n{'-'*40}\n{'-'*40}")

    def _save_trailing_stats(self):
        with open(self.trailing_stats_out_path, "wb") as f:
            pickle.dump(self.trailing_stats_dict, f)

    def log_validation_stats(self, hash2val_results):
        for h, val_dict in hash2val_results.items():
            name = self.hash2metadata[h]["name"]
            if self.tb_writer: 
                self.tb_writer.add_scalar(f"{name}/val_cost", val_dict["cost"], self.val_step)
                self.tb_writer.add_text(f"{name}/val_output", val_dict["text"], self.val_step)

        for h in self.trailing_stats_dict.keys():
            priority_queue = self.trailing_stats_dict[h]["best_sequence_priority_queue"]
            if len(priority_queue.queue)>0:
                name = self.hash2metadata[h]["name"]
                cost, best_sequence = priority_queue.peek_best()
                if self.tb_writer: 
                    self.tb_writer.add_text(f"{name}/best_sequence",
                                            f"best cost is: {cost}\n{best_sequence}",
                                            self.val_step)
                self._write_n_best(name=name, priority_queue=priority_queue)
                self._save_trailing_stats()


        self.val_step+=1

    def log_reference_baselines(self, bpe_strings: List[Tuple[str, str]]):
        for source_bpe_string, target_bpe_string in bpe_strings:

            h = hash_file(source_bpe_string)
            metadata = self.hash2metadata[h]
            cost_path_dict = make_cost_paths(host_path_to_volume=self.host_path_to_volume,
                                             container_path_to_volume=self.container_path_to_volume,
                                             volume_path_to_data=self.volume_path_to_data,
                                             volume_path_to_tmp=self.volume_path_to_tmp,
                                             data_path_to_target=metadata["base_asbly_path"],
                                             data_path_to_testcases=metadata["testcase_path"],
                                             assembly_name=metadata["name"])

            cost, failed_tunit, failed_cost = get_stoke_cost(bpe_string=target_bpe_string,
                                                             container_name=self.container_name,
                                                             cost_path_dict=cost_path_dict,
                                                             assembly_name=metadata["name"],
                                                             cost_conf=metadata["cost_conf"],
                                                             max_cost=self.max_score)

            effective_cost = min(cost, self.max_score)
            # get trailing stats for advantage
            self.hash2metadata[h]["reference_score"] = effective_cost

def get_stoke_cost(bpe_string: str,
                   container_name: str,
                   cost_path_dict: Dict[str, str],
                   assembly_name: str,
                   cost_conf,
                   max_cost = 9999) -> (float, bool, bool, Dict[str, str]):
    if not assembly_name: 
        breakpoint()
    formatted_string, _ = bpe2formatted(bpe_string, function_name = assembly_name, remove_footer = True)
    host_abs_path_raw_rewrite = cost_path_dict["host_abs_path_raw_rewrite"]
    host_abs_path_asbly_rewrite = cost_path_dict["host_abs_path_asbly_rewrite"]
    container_abs_path_raw_rewrite = cost_path_dict["container_abs_path_raw_rewrite"]
    container_abs_path_asbly_rewrite = cost_path_dict["container_abs_path_asbly_rewrite"]
    container_abs_path_to_functions = cost_path_dict["container_abs_path_to_functions"]
    container_abs_path_to_target = cost_path_dict["container_abs_path_to_target"]
    container_abs_path_to_testcases = cost_path_dict["container_abs_path_to_testcases"]

    with open(os.open(host_abs_path_raw_rewrite, os.O_CREAT | os.O_WRONLY, 0o777), "w+") as fh: # allows full permissions
        fh.write(formatted_string)
    tunit_rc, tunit_stdout = make_tunit_file(container_name=container_name,
                                             in_f=container_abs_path_raw_rewrite,
                                             out_f=host_abs_path_asbly_rewrite,
                                             fun_dir=container_abs_path_to_functions,
                                             live_dangerously=True)

    if tunit_rc == 0:

        cost_rc, cost_stdout, cost, correct = test_costfn(
            container_name = container_name,
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


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss
