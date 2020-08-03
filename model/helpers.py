# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
import hashlib
import subprocess
import re
import heapq
import json
from logging import Logger
from typing import Callable, Optional, List, Dict, Tuple
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
import yaml
from vocabulary import Vocabulary
from plotting import plot_heatmap
from os import makedirs
from time import time
from os.path import join, dirname

from typing import Union
from subproc import run
# monkey patch

subprocess.run(["sudo", "sleep", "3"])

subprocess.run = run

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

class PriorityQueue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.queue = []
        heapq.heapify(self.queue)

    def append(self, cost: int, sequence: str):
        if (-cost, sequence) in self.queue:
            return None, None
        elif len(self.queue) >= self.maxlen:
            neg_cost, worst_seq = heapq.heappushpop(self.queue, (-cost, sequence))
            return -neg_cost, worst_seq
        else:
            heapq.heappush(self.queue, (-cost, sequence))
            return None, None
    def peek_best(self):
        neg_cost, best_seq = sorted(self.queue, reverse=True)[0]
        return -neg_cost, best_seq
    def peek_worst(self):
        neg_cost, worst_seq = self.queue[0]
        return -neg_cost, worst_seq


def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)


def stitch_together(string):
    l = string.split()
    return ''.join(l).replace('▁', ' ').replace('</n>', '\n').replace('</->', '_')

def create_header_footer(assembly_string: str, function_name = None):
    if function_name == None:
        match = FUNCTION_NAME_REGEX.search(assembly_string)
        if match == None:
            print(assembly_string)
            function_name = 'default_function_name'
        else:
            function_name = match.group()
    header = f'''  .text\n  .global {function_name}\n  .type {function_name}, @function\n\n.{function_name}:\n'''
    footer = f"\n\n.size {function_name}, .-{function_name}"
    return header, footer


REMOVE_HEADER_REGEX = re.compile('(\s+.text\s*\n)|(\s+.global\s+[^\n]+\n)|(\s+.type\s+[^\n]+,\s+@function\s*\n\n)|(\s*\.[^\n:]+:\s*\n)')


def bpe2formatted(assembly_string: str, function_name = None, header_footer : Tuple[str, str] = None, remove_header: bool = True,
                            remove_footer: bool = True):
    un_bpe_string = stitch_together(assembly_string)
    #if re.search('(?<= \n nop \n nop)', un_bpe_string): 
    #    un_bpe_string = un_bpe_string[:re.search('(?<= \n nop \n nop)', un_bpe_string).start()]
    if remove_footer:
        un_bpe_string = REMOVE_FOOTER_REGEX.sub("", un_bpe_string)
    if remove_header:
        if not function_name: 
            breakpoint()
        assert function_name, "in order to strip the header, you must specify a funciton name"
        REMOVE_HEADER_REGEX.sub("", un_bpe_string)
    if not header_footer:
        header_footer = create_header_footer(assembly_string=un_bpe_string, function_name=function_name)
    return header_footer[0] + un_bpe_string + header_footer[1], header_footer


def hash_file(file_string: str, encoding: str = "utf-8") -> str:
    m = hashlib.sha512()
    m.update(bytes(file_string, encoding))
    return m.hexdigest()

def make_tunit_file(container_name: str, in_f: str, out_f: str, fun_dir: str, live_dangerously: bool = False):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    try:
        with open(out_f, "w") as f:
            tunit_proc = subprocess.run(
                ['sudo', 'docker', 'exec', container_name,
                 '/home/stoke/stoke/bin/stoke', 'debug', 'tunit', '--target', in_f,'--functions', fun_dir, "--prune", live_dangerously_str],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=25)

    except subprocess.TimeoutExpired as err:
        return -11747, err

    return tunit_proc.returncode, tunit_proc.stdout


def test_costfn(container_name: str,
                target_f: str,
                rewrite_f: str,
                testcases_f: str,
                fun_dir: str,
                settings_conf: Dict[str, str],
                live_dangerously = True):
    live_dangerously_str = "--live_dangerously" if live_dangerously else ""
    try:
        cost_test = subprocess.run(
            ['sudo', 'docker', 'exec', container_name,
             '/home/stoke/stoke/bin/stoke', 'debug', 'cost',
             '--target', target_f,
             '--rewrite', rewrite_f,
             '--testcases', testcases_f,
             '--functions', fun_dir,
             "--prune", live_dangerously_str,
             "--def_in", settings_conf["def_in"],
             "--live_out", settings_conf["live_out"],
             "--distance", settings_conf["distance"],
             "--misalign_penalty", str(settings_conf["misalign_penalty"]),
             "--sig_penalty", settings_conf["sig_penalty"],
             "--cost", settings_conf["costfn"],
             "--training_set", settings_conf["training_set"]] ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25)
        if cost_test.returncode == 0:
            cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
            correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()
        else:
            cost = -10701
            correct = "failed"
        return cost_test.returncode, cost_test.stdout, cost, correct

    except subprocess.TimeoutExpired as err:
        return -11785, err, -11785, "timeout"


def verify_rewrite(container_name: str,
                target_f: str,
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
            ['sudo', 'docker', 'exec', container_name,
             '/home/stoke/stoke/bin/stoke', 'debug', 'verify',
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
            timeout=25)
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

def make_verify_rewrite_paths(host_path_to_volume: str,
                               container_path_to_volume: str,
                               volume_path_to_tmp: str,
                               rewrite_id: str):

    machine_output_filename = rewrite_id + ".verify"

    host_abs_path_machine_output = join(host_path_to_volume, volume_path_to_tmp, machine_output_filename)
    container_abs_path_machine_output = join(container_path_to_volume, volume_path_to_tmp, machine_output_filename)

    return host_abs_path_machine_output, container_abs_path_machine_output


def verify_and_rewrite_testcase(container_name: str,
                                cost_path_dict: Dict[str, str],
                container_path_to_machine_output: str,
                host_path_to_machine_output: str, 
                settings_conf: Dict[str, str],
                new_testcase_idx: int,
                strategy: str = "hold_out",
                live_dangerously = True):
    container_path_to_target = cost_path_dict["container_abs_path_to_target"]
    container_path_to_rewrite = cost_path_dict["container_abs_path_asbly_rewrite"]
    container_path_to_testcases = cost_path_dict["container_abs_path_to_testcases"]
    container_path_to_functions = cost_path_dict["container_abs_path_to_functions"]
    host_path_to_testcases = cost_path_dict["host_abs_path_to_testcases"]

    verify_returncode =  verify_rewrite(container_name = container_name,
                                        target_f=container_path_to_target,
                                        rewrite_f=container_path_to_rewrite,
                                        testcases_f=container_path_to_testcases,
                                        fun_dir=container_path_to_functions,
                                        settings_conf=settings_conf,
                                        machine_output_f=container_path_to_machine_output,
                                        strategy=strategy,
                                        live_dangerously=live_dangerously)
    if verify_returncode == 0:
        is_verified_correct, counter_examples_available, counterexample_str = parse_verify_machine_output(host_path_to_machine_output)

        if is_verified_correct and counter_examples_available:
            add_counterexample_to_testcases(counterexample_str=counterexample_str,
                                            path_to_testcases=host_path_to_testcases,
                                            new_testcase_idx=new_testcase_idx)
    else:
        is_verified_correct = counter_examples_available = False


    return is_verified_correct, counter_examples_available


def make_cost_paths(host_path_to_volume: str,
                    container_path_to_volume: str,
                    volume_path_to_data: str,
                    volume_path_to_tmp: str,
                    data_path_to_target: str,
                    data_path_to_testcases: str,
                    assembly_name: str) -> Dict[str, str]:
    rewrite_id = (assembly_name + "_" + str(time())).replace(".", "_")
    return {"rewrite_id": rewrite_id,
    "host_abs_path_raw_rewrite": join(host_path_to_volume, volume_path_to_tmp, rewrite_id + ".tmp"),
    "host_abs_path_asbly_rewrite": join(host_path_to_volume, volume_path_to_tmp, rewrite_id + ".s"),
    "host_abs_path_to_testcases": join(host_path_to_volume, volume_path_to_data, data_path_to_testcases),
    "container_abs_path_raw_rewrite": join(container_path_to_volume, volume_path_to_tmp, rewrite_id + ".tmp"),
    "container_abs_path_asbly_rewrite": join(container_path_to_volume, volume_path_to_tmp, rewrite_id + ".s"),
    "container_abs_path_to_functions": dirname(join(container_path_to_volume, volume_path_to_data, data_path_to_target)),
    "container_abs_path_to_target": join(container_path_to_volume, volume_path_to_data, data_path_to_target),
    "container_abs_path_to_testcases": join(container_path_to_volume, volume_path_to_data, data_path_to_testcases)}



def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(log_file: str = None) -> Logger:
    """
    Create a logger for logging the training/testing process.

    :param log_file: path to file where log is stored as well
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.getLogger("").addHandler(sh)
    logger.info("Hello! This is Joey-NMT.")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  src_vocab: Vocabulary, trg_vocab: Vocabulary,
                  logging_function: Callable[[str], None]) -> None:
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
            len(train_data), len(valid_data),
            len(test_data) if test_data is not None else 0)

    logging_function("First training example:\n\t[SRC] %s\n\t[TRG] %s",
        " ".join(vars(train_data[0])['src']),
        " ".join(vars(train_data[0])['trg']))

    logging_function("First 10 words (src): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10])))
    logging_function("First 10 words (trg): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10])))

    logging_function("Number of Src words (types): %d", len(src_vocab))
    logging_function("Number of Trg words (types): %d", len(trg_vocab))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def store_attention_plots(attentions: np.array, targets: List[List[str]],
                          sources: List[List[str]],
                          output_prefix: str, indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = "{}.{}.pdf".format(output_prefix, i)
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                               row_labels=src, output_path=plot_file,
                               dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                                   row_labels=src, output_path=None, dpi=50)
                tb_writer.add_figure("attention/{}.".format(i), fig,
                                     global_step=steps)
        # pylint: disable=bare-except
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
