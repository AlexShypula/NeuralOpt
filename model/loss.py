# coding: utf-8
"""
Module to implement training loss
"""

import torch
import numpy as np
import re
from torch import nn, Tensor
from torch.autograd import Variable
from time import time
from os.path import join, basename, dirname
from helpers import mkdir, hash_file, make_tunit_file, test_costfn, bpe2formatted
from collections import deque
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Dict

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

class StokeCostManager:
    def __init__(self, hash2metadata, tmp_folder_path, data_path, tb_writer, max_len = 256,
                 max_score = 9999, n_workers=8):
        mkdir(tmp_folder_path)
        self.tmp_folder_path = tmp_folder_path
        self.data_path = data_path
        self.tb_writer = tb_writer
        self.hash2metadata = hash2metadata
        self.n_workers = n_workers
        self.trailing_stats_dict = dict()
        self.val_step = 0
        for h in hash2metadata.keys():
            self.trailing_stats_dict[h] = {"costs": deque(maxlen = max_len),
                                            "failed_tunit": deque(maxlen = max_len),
                                            "failed_cost": deque(maxlen = max_len),
                                            "normalized_advantage": deque(maxlen = max_len),
                                            "n_steps": 0}
            self.hash2metadata[h]["name"] = basename(self.hash2metadata[h]["base_asbly_path"])
        self.max_score = max_score

    def get_rl_cost(self, source_bpe_string: str, hypothesis_bpe_string: str):
        h = hash_file(source_bpe_string)
        metadata = self.hash2metadata[h]
        cost_conf = metadata["cost_conf"]
        target_asbly_path = join(self.data_path, metadata["base_asbly_path"])
        testcase_path = join(self.data_path, metadata["testcase_path"])
        name = metadata["name"]
        cost, failed_tunit, failed_cost = get_stoke_cost(bpe_string=hypothesis_bpe_string,
                                                            orig_file_path=target_asbly_path,
                                                            testcase_path = testcase_path,
                                                            tmp_folder=self.tmp_folder_path,
                                                            name = name,
                                                            cost_conf = cost_conf,
                                                            max_cost=self.max_score)
        effective_cost = min(cost, self.max_score)
        # get trailing stats for advantage
        cost_std = 1 if len(self.trailing_stats_dict[h]["costs"]) == 0 else np.std(self.trailing_stats_dict[h]["costs"])
        cost_mean = 0 if len(self.trailing_stats_dict[h]["costs"]) == 0 else np.mean(self.trailing_stats_dict[h]["costs"])

        normalized_advantage = (effective_cost - cost_mean) / (cost_std if cost_std != 0 else 1)

        return h, normalized_advantage, {"normalized_advantage": normalized_advantage,
                                         "cost": effective_cost,
                                         "failed_tunit": failed_tunit,
                                         "failed_cost": failed_cost}

    def get_rl_cost_wrapper(self, args):
        return self.get_rl_cost(**args)

    def parallel_get_rl_cost(self, bpe_strings: List[Tuple[str, str]]):
        arg_list = []
        for (source_bpe_str, hypothesis_bpe_str) in bpe_strings:
            arg_list.append({"source_bpe_string": source_bpe_str, "hypothesis_bpe_string": hypothesis_bpe_str})
        #breakpoint()
        hash_cost_list = list(map(self.get_rl_cost_wrapper, arg_list))
#list(ThreadPool(self.n_workers).map(self.get_rl_cost_wrapper, arg_list))
        return hash_cost_list

    def update_buffers(self, hash_stats_list: Tuple[str, Dict]):

        batch_cost = 0
        for h, stats in hash_stats_list:
            normalized_advantage = stats["normalized_advantage"]
            effective_cost = stats["cost"]
            failed_tunit = stats["failed_tunit"]
            failed_cost = stats["failed_cost"]

            # update the buffers
            self.trailing_stats_dict[h]["normalized_advantage"].append(normalized_advantage)
            self.trailing_stats_dict[h]["costs"].append(effective_cost)
            self.trailing_stats_dict[h]["failed_tunit"].append(failed_tunit)
            self.trailing_stats_dict[h]["failed_cost"].append(failed_cost)

            batch_cost += effective_cost
        batch_cost /= len(hash_stats_list)
        return batch_cost

    def log_buffer_stats(self):

        for h in self.trailing_stats_dict.keys():
            name = self.hash2metadata[h]["name"]

            # re-calculate stats for logger
            mean_normalized_advantage = np.mean(self.trailing_stats_dict[h]["normalized_advantage"])
            trailing_cost_mean = np.mean(self.trailing_stats_dict[h]["costs"])
            trailing_cost_std = np.std(self.trailing_stats_dict[h]["costs"])
            trailing_failed_tunit = np.mean(self.trailing_stats_dict[h]["failed_tunit"])
            trailing_failed_cost = np.mean(self.trailing_stats_dict[h]["failed_cost"])
            step_no = self.trailing_stats_dict[h]["n_steps"]
            self.trailing_stats_dict[h]["n_steps"] += 1

            self.tb_writer.add_scalar(f"{name}/normalized_advantage", mean_normalized_advantage, step_no)
            self.tb_writer.add_scalar(f"{name}/trailing_cost", trailing_cost_mean, step_no)
            self.tb_writer.add_scalar(f"{name}/trailing_std", trailing_cost_std, step_no)
            self.tb_writer.add_scalar(f"{name}/trailing_failed_tunit", trailing_failed_tunit, step_no)
            self.tb_writer.add_scalar(f"{name}/trailing_failed_cost", trailing_failed_cost, step_no)

    def log_validation_stats(self, hash2val_results):
        for h, val_dict in hash2val_results.items():
            name = self.hash2metadata[h]["name"]
            self.tb_writer.add_scalar(f"{name}/val_cost", val_dict["cost"], self.val_step)
            self.tb_writer.add_text(f"{name}/val_output", val_dict["text"], self.val_step)

        self.val_step+=1






def get_stoke_cost(bpe_string: str,
                   orig_file_path: str,
                   testcase_path: str,
                   tmp_folder: str,
                   name: str,
                   cost_conf,
                   max_cost = 9999) -> float:
    formatted_string, _ = bpe2formatted(bpe_string, remove_footer = True)
    tmp_name = name + str(time())
    raw_path = join(tmp_folder, tmp_name + ".tmp")
    asbly_path = join(tmp_folder, tmp_name + ".s")
    fun_dir = dirname(orig_file_path)
    with open(raw_path, "w") as fh:
        fh.write(formatted_string)
    tunit_rc, tunit_stdout = make_tunit_file(raw_path,
                                             asbly_path,
                                             fun_dir,
                                             live_dangerously=True)

    if tunit_rc == 0:

        cost_rc, cost_stdout, cost, correct = test_costfn(
            target_f=orig_file_path,
            rewrite_f=asbly_path,
            testcases_f=testcase_path,
            fun_dir=fun_dir,
            settings_conf=cost_conf,
            live_dangerously=True)

    tunit_failed = False if tunit_rc == 0 else True
    cost_failed = False if tunit_rc == 0 and cost_rc == 0 else True

    if tunit_rc == 0 and cost_rc == 0:
        return float(cost), tunit_failed, cost_failed
    else:
        return float(max_cost), tunit_failed, cost_failed

# def parallel_stoke_cost(bpe_strings: List[str],
#                         orig_file_paths: List[str],
#                         fun_dirs: List[str],
#                         tmp_folder: str,
#                         live_dangerously: bool = True,
#                         n_workers: int = 1):
#     assert len(bpe_strings) == len(orig_file_paths) == len(fun_dirs), "cost list has different lengths of args"
#
#     jobs = []
#     for i in range(len(bpe_strings)):
#         job = {"bpe_string": bpe_strings[i],
#                "orig_file_path":orig_file_paths[i],
#                "tmp_folder": tmp_folder,
#                "fun_dir": fun_dirs[i],
#                "live_dangerously": live_dangerously}
#         jobs.append(job)
#
#     cost_list = list(ThreadPool(n_workers).map(get_stoke_cost_parallel, jobs))
#     return cost_list
#
# def get_stoke_cost_parallel(args_dict):
# 	return get_stoke_cost(**args_dict)

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
