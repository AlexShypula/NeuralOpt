# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

import sacrebleu
from typing import List
# from .utils.stoke_test_costfn import bpe2formatted, make_tunit_file, test_costfn
# from time import time
# from os.path import join
# from multiprocessing.pool import ThreadPool
# from typing import Union


# def parallel_stoke_cost(bpe_strings:List[str],
#                         orig_file_paths: List[str],
#                         fun_dirs:List[str],
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

# def get_stoke_cost(bpe_string: str, orig_file_path: str, testcases_path: str,
#                    tmp_folder: str, fun_dir: str, live_dangerously = True,
#                    max_cost: Union[int,float] = 3000.0) -> float:
#     formatted_string, _ = bpe2formatted(bpe_string, remove_footer=True) # second return value is header footer tuple
#     tmp_name = str(time())
#     raw_path = join(tmp_folder, tmp_name + ".raw")
#     asbly_path = join(tmp_folder, tmp_name + ".s")
#     with open(raw_path, "w") as fh:
#         fh.write(formatted_string)
#     # args are : (in_f: str, out_f: str, fun_dir: str, live_dangerously: bool = False):
#     tunit_rc, tunit_stdout = make_tunit_file(in_f=raw_path,
#                                              out_f=asbly_path,
#                                              fun_dir=fun_dir,
#                                              live_dangerously=live_dangerously)
#     if tunit_rc == 0:
#         # args are: (target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, live_dangerously: bool = False):
#         cost_rc, cost_stdout, cost, correct = test_costfn(
#             target=orig_file_path,
#             rewrite_f=asbly_path,
#             testcases_f=testcases_path,
#             fun_dir=fun_dir,
#             live_dangerously=live_dangerously)
#
#     if cost_rc == 0 and tunit_rc == 0:
#         return float(cost)
#     else:
#         return float(max_cost)
#

def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references)


def bleu(hypotheses, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.raw_corpus_bleu(sys_stream=hypotheses,
                                     ref_streams=[references]).score

def sent_bleu(hypotheses, references)-> List[float]:
    scores = []
    for h, r in zip(hypotheses, references):
        scores.append(sacrebleu.sentence_bleu(hypothesis = h, references = r).score)
    return scores



def token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    def split_by_space(string):
        """
        Helper method to split the input based on spaces.
        Follows the same structure as list(inp)
        :param string: string
        :return: list of strings
        """
        return string.split(" ")

    correct_tokens = 0
    all_tokens = 0
    split_func = split_by_space if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(split_func(hyp), split_func(ref)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (hyp, ref) in zip(hypotheses, references)
                             if hyp == ref])
    return (correct_sequences / len(hypotheses))*100 if hypotheses else 0.0
