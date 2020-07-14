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

def create_header_footer(assembly_string: str):
    match = FUNCTION_NAME_REGEX.search(assembly_string)
    if match == None:
        print(assembly_string)
        function_name = 'NA'
    else:
        function_name = match.group()
    header = f'''  .text\n  .global {function_name}\n  .type {function_name}, @function\n\n'''
    footer = f".size {function_name}, .-{function_name}"
    return header, footer

def bpe2formatted(assembly_string: str, header_footer : Tuple[str, str] = None, remove_footer: bool = False):
    # header is the zeroth indexed value in header_footer, and footer should be the fist indexed value
    un_bpe_string = stitch_together(assembly_string)
    if remove_footer:
        un_bpe_string = REMOVE_FOOTER_REGEX.sub("", un_bpe_string)
    if not header_footer:
        header_footer = create_header_footer(un_bpe_string)
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
             "--misalign_penalty", settings_conf["misalign_penalty"],
             "--sig_penalty", settings_conf["sig_penalty"],
             "--cost", settings_conf["costfn"]] ,
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
