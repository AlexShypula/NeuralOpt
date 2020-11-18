# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
from typing import List, Optional
from logging import Logger
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import random

import torch
from torchtext.data import Dataset, Field

from helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint, store_attention_plots, bpe2formatted, mkdir
from metrics import bleu, chrf, token_accuracy, sequence_accuracy
from modeling import build_model, Model
from batch import Batch
from data import load_data, make_data_iter, MonoDataset
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from vocabulary import Vocabulary
from loss import StokeCostManager
from tqdm import tqdm
from os.path import join, basename, splitext
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


os.environ["CUDA_VISIBLE_DEVICES"]="0"

@dataclass
class ParseOptions:
    cfg_file : str = field(metadata=dict(args=["-config", "--path_to_config"]))
    ckpt: str = field(metadata=dict(args=["-ckpt", "--path_to_checkpoint"]))
    n_best: int = field(metadata=dict(args=["-nbest", "--nbest_beams"]), default=None)
    beam_size: int = field(metadata=dict(args=["-beam_size", "--beam_size"]), default=None)
    beam_alpha: float = field(metadata=dict(args=["-beam_alpha", "--beam_alpha"]), default=None)
    output_path: str = field(metadata=dict(args=["-output_path", "--output_path"]), default=None)
    debug: bool = field(metadata=dict(args=["-d", "--debug"]), default = False)
    exp_name: str = field(metadata=dict(args=["-exp_name", "--experiment_name"]), default="")
    api_ip_addr: str = field(metadata=dict(args=["-api_ip_addr", "--api_ip_address"]), default = "127.0.0.1")
    # the following should be colon, ":" separated and in "train, dev, test"
    datasets_to_test: str = field(metadata=dict(args=["-data_to_test", "--datasets_to_test"]), default = "test")



CSV_KEYS = ["name", "cost", "O0_cost", "Og_cost", "rc", "failed_cost", "base_asbly_path"]


rc2axis = {-1: "failed",
           0: "incorrect",
           1: "worse than -O0",
           2: "matched -O0",
           3: "between -O0 and -Og",
           4: "matched -Og",
           5: "better than -Og"}

# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset,
                     logger: Logger,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: Optional[str],
                     cost_manager: StokeCostManager = None,
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 1, beam_alpha: int = -1,
                     batch_type: str = "sentence",
                     n_best: int  = 0
                     ) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param logger: logger
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    assert n_best <= beam_size
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_lengths()

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores = model.run_batch(
                batch=batch, beam_size=beam_size, beam_alpha=beam_alpha,
                max_output_length=max_output_length, n_best = n_best)

            # sort outputs back to original order
            if n_best >= 1: 
                all_outputs.extend([output[i] for i in sort_reverse_index])
            else: 
                all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])

        assert len(all_outputs) == len(data)

        if loss_function is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]


        # post-process
        if level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v)
                                for v in valid_references]
            #valid_hypotheses = [bpe_postprocess(v) for

        rc_stats_dict = {}
        for i in range(-1, 6):
            rc_stats_dict[i] = {"counts": 0, "costs": []}
        current_valid_score = 0
        decoded_valid = None
        valid_attention_scores = None

        if n_best > 1:
            valid_beams = []
            for beams in all_outputs:
                beams = model.trg_vocab.arrays_to_sentences(arrays=beams,
                                                            cut_at_eos=True)
                beams = [join_char.join(t) for t in beams]
                if level == "bpe":
                    beams = [bpe_postprocess(t) for t in beams]
                valid_beams.append(beams)
            valid_hypotheses = []
            individual_record_list = []
            pbar = tqdm(total = len(valid_beams), smoothing=0, desc="evaluating all beams with STOKE")
            for source, hypotheses in zip(valid_sources, valid_beams):
                rc, hypothesis_string, stats, comparison_string, metadata = cost_manager.eval_beams_v2(source, hypotheses)
                rc_stats_dict[rc]["counts"]+=1
                rc_stats_dict[rc]["costs"].append(stats["cost"])
                valid_hypotheses.append(hypothesis_string)
                individual_record_list.append({**metadata, **stats,
                                               "comparison_string": comparison_string,
                                               "rc": rc, "hypothesis_string": hypothesis_string})
                pbar.update()
            for rc, stats in rc_stats_dict.items():
                rc_stats_dict[rc]["mean_cost"] = np.mean(stats["costs"]) if len(stats["costs"]) > 0 else -1
                rc_stats_dict[rc]["std"] = np.std(stats["costs"]) if len(stats["costs"]) > 1 else 0

            n_best_results = {"return_code_stats": rc_stats_dict, "individual_records": individual_record_list}

        else:

            # decode back to symbols
            decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                                cut_at_eos=True)

            valid_hypotheses = [join_char.join(t) for t in decoded_valid]

            if level == "bpe":
                valid_hypotheses = [bpe_postprocess(v) for v in valid_hypotheses]


            if n_best == 1:
                individual_record_list = []
                result_tuples = cost_manager.batch_eval_greedy(zip(valid_sources, valid_hypotheses))
                valid_hypotheses = []
                for rc, hypothesis_string, stats, comparison_string, metadata in result_tuples:
                    rc_stats_dict[rc]["counts"]+=1
                    rc_stats_dict[rc]["costs"].append(stats["cost"])
                    valid_hypotheses.append(hypothesis_string)
                    individual_record_list.append({**metadata, **stats,
                                                   "comparison_string": comparison_string,
                                                   "rc": rc, "hypothesis_string": hypothesis_string})
                for rc, stats in rc_stats_dict.items():
                    rc_stats_dict[rc]["mean_cost"] = np.mean(stats["costs"]) if len(stats["costs"]) > 0 else -1
                    rc_stats_dict[rc]["std"] = np.std(stats["costs"]) if len(stats["costs"]) > 0 else -1
                n_best_results = {"return_code_stats": rc_stats_dict, "individual_records": individual_record_list}

            else:
                n_best_results = {}

                # if references are given, evaluate against them
                if eval_metric.lower() == "stoke":

                    hashes_advantages_stats = cost_manager.parallel_get_rl_cost(zip(valid_sources, valid_hypotheses))
                    hash2val_results = {}
                    c = 0
                    for hyp, (h, advantages, stats) in zip(valid_hypotheses, hashes_advantages_stats):
                        hash2val_results[h] = {"cost": stats["cost"],
                                           "text": bpe2formatted(assembly_string=hyp, function_name = "default_validation_name", remove_footer=True)[0]}
                        c += stats["cost"]
                    current_valid_score = c / len(valid_hypotheses)
                    cost_manager.log_validation_stats(hash2val_results)

                elif valid_references:
                    assert len(valid_hypotheses) == len(valid_references)

                    current_valid_score = 0

                    if eval_metric.lower() == 'bleu':
                        # this version does not use any tokenization
                        current_valid_score = bleu(valid_hypotheses, valid_references)
                    elif eval_metric.lower() == 'chrf':
                        current_valid_score = chrf(valid_hypotheses, valid_references)
                    elif eval_metric.lower() == 'token_accuracy':
                        current_valid_score = token_accuracy(
                            valid_hypotheses, valid_references, level=level)
                    elif eval_metric.lower() == 'sequence_accuracy':
                        current_valid_score = sequence_accuracy(
                            valid_hypotheses, valid_references)
                else:
                    current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores, n_best_results


# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str,
         exp_name: str = "",
         api_ip_addr: str = "127.0.0.1",
         n_best: int = 0,
         beam_size: int = None,
         beam_alpha: float = None,
         output_path: str = None,
         debug: bool = False, 
         save_attention: bool = False,
         logger: Logger = None,
         datasets_to_test: str = "test") -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """
    for data_name in datasets_to_test.split(":"):
        assert data_name in ("train", "dev", "test")

    if logger is None:
        logger = make_logger()

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    model_dir = cfg["training"]["model_dir"]
    if ckpt is None:
        #model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"


    data_config = cfg["data"]

    with open(data_config.get("hash2metadata")) as fh:
        hash2metadata = json.load(fh)
    # cost_manager

    max_score = data_config.get("max_score", 9999)
    asm_names_to_save = data_config.get("asm_names_to_save")
    asm_names_to_save = asm_names_to_save.split(":") if asm_names_to_save else []
    cost_manager = StokeCostManager(hash2metadata = hash2metadata,
                                         #container_name = data_config.get("container_name"),
                                         host_path_to_volume = data_config.get("host_path_to_volume"),
                                         container_path_to_volume = data_config.get("container_path_to_volume"),
                                         volume_path_to_data = data_config.get("volume_path_to_data"),
                                         volume_path_to_tmp = data_config.get("volume_path_to_tmp"),
                                         tb_writer = None,
                                         n_best_seq_dir = None, 
                                         trailing_stats_out_path="{}/trailing_stats.pkl".format(model_dir),
                                         baseline_cost_key= data_config.get("baseline_cost_key", "O0_cost"),  
                                         asm_names_to_save = None,
                                         verifiction_strategy = data_config.get("verification_strategy", "hold_out"),
                                         new_testcase_beginning_index = data_config.get(
                                            "new_testcase_beginning_index", 2000),
                                         max_len = data_config.get("max_len"),
                                         max_score = data_config.get("max_score"),
                                         n_workers = data_config.get("n_workers"),
                                         keep_n_best_seqs=data_config.get("keep_n_best_seqs", 10),
                                         api_ip_adddr=api_ip_addr,
                                         container_port=data_config.get("container_port", 6000),
                                         trailing_stats_in_path=data_config.get("trailing_stats_in_path")
                                         )


    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    ## TODO: see if you can subset the train-data here to evaluate on a subset
    torch.manual_seed(0)
    random.seed(0)

    train_data, dev_data, test_data, src_vocab, trg_vocab, src_field, trg_field = load_data(
        data_cfg=cfg["data"])
    train_data, _ = train_data.split(split_ratio = [0.03, 1 - 0.03], random_state = random.getstate())
    if debug: 
        keep, _ = dev_data.split(
                            split_ratio=[0.1, 1 - 0.1],
                                        random_state=random.getstate())
        dev_data = keep

        keep, _ = test_data.split(
                            split_ratio=[0.1, 1 - 0.1],
                                        random_state=random.getstate())
        test_data = keep

    all_data_dict = {"train": train_data, "dev": dev_data, "test": test_data}
    data_to_predict = {data: all_data_dict[data] for data in datasets_to_test.split(":")}

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if not beam_size:
        if "testing" in cfg.keys():
            beam_size = cfg["testing"].get("beam_size", 1)
        else:
            beam_size = 1
    if not beam_alpha:
        if "testing" in cfg.keys():
            beam_alpha = cfg["testing"].get("beam_alpha", -1)
        else:
            beam_alpha = -1
    if not n_best:
        if "testing" in cfg.keys():
            n_best = cfg["testing"].get("n-best", -1)
        else:
            n_best= 1

    assert beam_size >= n_best

    for data_set_name, data_set in data_to_predict.items():

        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores, results = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha, logger=logger, n_best=n_best, cost_manager = cost_manager)
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            decoding_description = "Greedy decoding" if beam_size < 2 else \
                "Beam search decoding with beam size = {} and alpha = {}".\
                    format(beam_size, beam_alpha)
            logger.info("%4s %s: %6.2f [%s]",
                        data_set_name, eval_metric, score, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)

        if save_attention:
            if attention_scores:
                attention_name = "{}.{}.att".format(data_set_name, step)
                attention_path = os.path.join(model_dir, attention_name)
                logger.info("Saving attention plots. This might take a while..")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=data_set.src,
                                      indices=range(len(hypotheses)),
                                      output_prefix=attention_path)
                logger.info("Attention plots saved to: %s", attention_path)
            else:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not available "
                               "when using beam search. "
                               "Set beam_size to 1 for greedy decoding.")
        output_path = output_path if output_path else cfg["training"]["model_dir"]
        ckpt_name = splitext(basename(ckpt))[0]
        exp_ckpt_name = "{}_{}".format(exp_name, ckpt_name) if exp_name != "" else ckpt_name
        output_dir = "{}/{}".format(output_path,exp_ckpt_name)
        mkdir(output_dir)
        if n_best > 0: 
            return_code_stats = results["return_code_stats"]
            individual_records = results["individual_records"]
            comparison_str_dir = join(output_dir, "comparisons_{}".format(data_set_name))
            mkdir(comparison_str_dir)
            with open(join(output_dir, "stats_{}.csv".format(data_set_name)), "w") as csv_fh:
                csv_writer = csv.DictWriter(f = csv_fh, fieldnames=CSV_KEYS)
                csv_writer.writeheader()
                for individual_record in individual_records:
                    csv_writer.writerow({key: individual_record[key] for key in CSV_KEYS})
                    with open(join(comparison_str_dir, f"{individual_record['name']}.comparison"), "w") as fh:
                        fh.write(individual_record["comparison_string"])
            percentage_dict = {}
            cost_dict = {}
            stdv_dict = {}
            total_cts = 0
            for d in return_code_stats.values():
                total_cts+= d["counts"]
            for rc, d in return_code_stats.items():
                title = rc2axis[rc]
                percentage_dict[title] = (d["counts"] / total_cts) * 100
                # if they did assemble
                if rc > -1:
                    cost_dict[title] = d["mean_cost"]
                    stdv_dict[title] = d["std"]

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
            plt.title("Experiment: {} on {} Distribution of Result Types".format(exp_ckpt_name, data_set_name))
            plt.ylabel("Percentage")
            plt.xticks(rotation=75)
            plt.savefig(join(output_dir, "percentages_{}.png".format(data_set_name)), dpi=300, pad_inches=2, bbox_inches="tight")
            plt.clf()

            # COST DICT
            max_val = max(cost_dict.values())

            plt.rc('axes', axisbelow=True)
            plt.rcParams["font.family"] = "sans-serif"
            plt.grid(color='gray', linestyle='dashed')

            plt.bar(cost_dict.keys(), cost_dict.values(), color="darkgreen")
            plt.title("Experiment: {} on {} Average Stoke Cost by Performance Bucket".format(exp_ckpt_name, data_set_name))
            plt.ylabel("Stoke Cost")
            plt.yscale("log")
            plt.xticks(rotation=75)

            annotation_dict = {k: (f'{v:.2f}' if v != -1 else "NA") for k, v in cost_dict.items()}
            textstr = "Costs per Bucket:" + "\n\n" + '\n'.join([f"{k} = {v}" for k, v in annotation_dict.items()])
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            plt.text(6, max_val, textstr, fontsize=12,
                     verticalalignment='top', bbox=props)

            plt.savefig(join(output_path, "cost_{}.png",format(data_set_name)), dpi=300, pad_inches=2, bbox_inches="tight")
            plt.clf()

            # STDV DICT

            max_val = max(stdv_dict.values())

            plt.rc('axes', axisbelow=True)
            plt.rcParams["font.family"] = "sans-serif"
            plt.grid(color='gray', linestyle='dashed')

            plt.bar(stdv_dict.keys(), stdv_dict.values(), color="darkgreen")
            plt.title("Experiment: {} on {} Standard Deviation of Stoke Cost by Performance Bucket".\
                      format(exp_ckpt_name, data_set_name))
            plt.ylabel("Standard Deviation of Stoke Cost")
            plt.yscale("log")
            plt.xticks(rotation=75)

            annotation_dict = {k: (f'{v:.2f}' if v != -1 else "NA") for k, v in stdv_dict.items()}
            textstr = "Cost Standard Deviation per Bucket:" + "\n\n" + '\n'.join([f"{k} = {v}" for k, v in annotation_dict.items()])
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            plt.text(6, max_val, textstr, fontsize=12,
                     verticalalignment='top', bbox=props)
            plt.savefig(join(output_path, "stdev_{}.png".format(output_dir)), dpi=300, pad_inches=2, bbox_inches="tight")
            plt.clf()


        hyp_file = join(output_path, "hyps.txt")
        with open(hyp_file, mode="w", encoding="utf-8") as out_file:
            for hyp in hypotheses:
                out_file.write(hyp + "\n")
        logger.info("Translations saved to: %s", hyp_file)


def translate(cfg_file, ckpt: str, output_path: str = None) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name+tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix,
                                field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    logger = make_logger()

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha, logger=logger)
        return hypotheses

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # read vocabs
    src_vocab_file = cfg["data"].get(
        "src_vocab", cfg["training"]["model_dir"] + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get(
        "trg_vocab", cfg["training"]["model_dir"] + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    def tok_fun(s):
        if level == "char":
            return list(s)
        else:
            return s.split()
    # tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, <2: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 1
        beam_alpha = -1

    if not sys.stdin.isatty():
        # input file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            # write to outputfile if given
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s.", output_path_set)
        else:
            # print to stdout
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        batch_type = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    test(**vars(args))
