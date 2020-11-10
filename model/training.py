# coding: utf-8

"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import os
import queue
import json
import math
import dill
import queue

# import math
import numpy as np
#import multiprocessing_on_dill as mp
#import multiprocessing as mp
from torch import multiprocessing as mp
#from multiprocessing.queues import SimpleQueue
#from prwlock import RWLock
#from readerwriterlock import rwlock
mp.set_start_method('spawn', force = True)

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
from modeling import build_model
from batch import Batch, LearnerBatch
from helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, bpe_postprocess, BucketReplayBuffer, StopWatch
from modeling import Model
from prediction import validate_on_data
from loss import XentLoss, StokeCostManager, log_probs_and_entropy
from data import load_data, make_data_iter, shard_data
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from prediction import test
from actor import actor, actor_wrapper

from tqdm import tqdm
from typing import Dict
import gc
from torchtext.data import Field
from collections import deque
from copy import deepcopy
from fairseq import pdb

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"rlimit is {rlimit}")
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


os.environ["CUDA_VISIBLE_DEVICES"]="2"
torch.set_num_threads(8)

def init(l, model_id, ctr):
    global model_lock
    global latest_model_id
    global running_starts_counter
    model_lock = l
    latest_model_id = model_id
    running_starts_counter = ctr

# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: Model, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]
        data_config = config["data"]

        with open(data_config.get("hash2metadata")) as fh:
            self.hash2metadata = json.load(fh)

        # augment hash2metadata to contain the hash within it
        for h, metadata in self.hash2metadata.items():
            if "hash" not in metadata.keys():
                self.hash2metadata[h]["hash"] = h

        # files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get(
                                            "overwrite", False))
        self.logger = make_logger("{}/train.log".format(self.model_dir))
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(
            log_dir=self.model_dir + "/tensorboard/")

        # cost_manager
        self.max_score = data_config.get("max_score", 9999)
        asm_names_to_save = data_config.get("asm_names_to_save")
        asm_names_to_save = asm_names_to_save.split(":") if asm_names_to_save else []
        self.cost_manager = StokeCostManager(hash2metadata = self.hash2metadata,
                                             #container_name = data_config.get("container_name"),
                                             host_path_to_volume = data_config.get("host_path_to_volume"),
                                             container_path_to_volume = data_config.get("container_path_to_volume"),
                                             volume_path_to_data = data_config.get("volume_path_to_data"),
                                             volume_path_to_tmp = data_config.get("volume_path_to_tmp"),
                                             tb_writer = self.tb_writer,
                                             n_best_seq_dir="{}/best_seqs/".format(self.model_dir),
                                             trailing_stats_out_path="{}/trailing_stats.pkl".format(self.model_dir),
                                             baseline_cost_key= data_config.get("baseline_cost_key", "O0_cost"),  
                                             asm_names_to_save = asm_names_to_save,
                                             verifiction_strategy = data_config.get("verification_strategy", "hold_out"),
                                             new_testcase_beginning_index = data_config.get(
                                                "new_testcase_beginning_index", 2000),
                                             max_len = data_config.get("max_len"),
                                             max_score = data_config.get("max_score"),
                                             n_workers = data_config.get("n_workers"),
                                             keep_n_best_seqs=data_config.get("keep_n_best_seqs", 10),
                                             container_port=data_config.get("container_port", 6000),
                                             trailing_stats_in_path=data_config.get("trailing_stats_in_path")
                                             )

        #actor-learner
        self.n_actors = train_config.get("n_actors", 1)
        self.actor_devices = train_config.get("actor_devices", "cpu").split("/")
        self.learner_device = train_config.get("learner_device", "cuda:0")
        self.replay_buffer_size = train_config.get("replay_buffer_size", 512)
        self.save_learner_every = train_config.get("save_learner_every", 1)
        self.synchronized_al = train_config.get("synchronized_al", False)
        
        valid_freq = train_config.get("validation_freq", 1000)
        self.log_best_seq_stats_every = train_config.get("log_best_seq_stats_every", valid_freq)
        self.n_updates = train_config.get("n_updates", 0)
        self.bucket_buffer_splits = train_config.get("n_buffer_splits", 4)
        self.ppo_flag = train_config.get("ppo_flag", True)
        self.ppo_epsilon = train_config.get("ppo_epsilon", 0.2)
        #actor-learner required data
        self.shard_data = data_config.get("shard_data", True)
        self.shard_path = data_config.get("shard_path", None)
        self.use_shards = data_config.get("use_shards", True)
        if self.shard_data or self.use_shards:
            assert self.shard_path, "if sharding the data or using shards, a shard path must be specified"
        self.src_lang = data_config["src"]
        self.tgt_lang = data_config["trg"]
        self.train_path = data_config["train"]
        self.dev_path = data_config["dev"]
        self.test_path = data_config.get("test", None)
        self.level = data_config["level"]
        self.lowercase = data_config["lowercase"]
        self.max_sent_length = data_config["max_sent_length"]
        self.container_port = data_config.get("container_port", 6000)
        self.learner_model_path = "{}/learner.ckpt".format(self.model_dir)
        self.learner_tmp_path = "{}/tmp.ckpt".format(self.model_dir)





        # model
        self.model = model
        self.model_cfg = config["model"]
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.eos_index = self.model.eos_index
        self._log_parameters_list()

        # objective
        self.beta_entropy = train_config.get("beta_entropy", 0.0)
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.loss = None #XentLoss(pad_index=self.pad_index, smoothing=self.label_smoothing)
        self.normalization = train_config.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = queue.Queue(
            maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ['bleu',
                                    'chrf',
                                    'token_accuracy',
                                    'sequence_accuracy', 
                                    'stoke']:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', "
                                     "'token_accuracy', 'sequence_accuracy'.")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in ["ppl", "loss", "stoke"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf"]:
                self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric'.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size",
                                                self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",
                                                self.batch_type)

        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        self.current_batch_multiplier = self.batch_multiplier
        self.sentence_samples = train_config.get("sentence_samples", 1)

        #running starts
        self.no_running_starts = train_config.get("no_running_starts", 0)
        self.running_starts_multiplier = train_config.get("running_starts_multiplier", 1)
        self.running_starts_batch_size = train_config.get("running_starts_batch_size", self.batch_size)
        self.running_starts_batch_type = train_config.get("running_starts_batch_type", self.batch_type)

        # init attributes for batch training
        self.log_batch_score = 0
        self.multi_batch_loss = 0
        self.multi_batch_score = 0
        self.multi_batch_pct_failure = 0
        self.multi_batch_entropy = 0
        self.multi_batch_hash_list = []
        # self.epoch_loss = 0 already initialized in the epoch loop
        self.update = False
        #self.count = self.current_batch_multiplier - 1

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            #self.loss.cuda()

        # initialize accumalted batch loss (needed for batch_multiplier)
        self.norm_batch_loss_accumulated = 0
        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        #self.is_best = lambda score: score < self.best_ckpt_score \
        #    if self.minimize_metric else score > self.best_ckpt_score

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", True)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(model_load_path,
                                      reset_best_ckpt=reset_best_ckpt,
                                      reset_scheduler=reset_scheduler,
                                      reset_optimizer=reset_optimizer)
    def is_best(self, score): 
        if self.minimize_metric: 
            return score < self.best_ckpt_score
        else: 
            return score > self.best_ckpt_score
    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning("Wanted to delete old checkpoint %s but "
                                    "file does not exist.", to_delete)

        self.ckpt_queue.put(model_path)

        best_path = "{}/best.ckpt".format(self.model_dir)
        try:
            # create/modify symbolic link for best checkpoint
            symlink_update("{}.ckpt".format(self.steps), best_path)
        except OSError:
            # overwrite best.ckpt
            torch.save(state, best_path)

    def _save_learner(self) -> None:

        #assert latest_model_id and model_lock,"you need to have declared latest_model_id and model_lock to save learner"
        #assert type(latest_model_id.value) == int, "need to init a mp.Value object as latest model id with int value"
        # these are global objects initialized in train_and_validate_actor_learner
        
        #model_lock.acquire()
        #with model_lock.gen_wlock(), latest_model_id.get_lock():
        state = {"model_state": self.model.state_dict()}
        torch.save(state, self.learner_tmp_path) 
        model_lock.acquire()
        shutil.copy2(self.learner_tmp_path , self.learner_model_path)
        #latest_model_id.value += 1
        model_lock.release()

    def init_from_checkpoint(self, path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statement

    def _do_running_starts(self, train_data: Dataset) -> None:

        running_starts_iter = make_data_iter(train_data,
                                             batch_size=self.running_starts_batch_size,
                                             batch_type=self.running_starts_batch_type,
                                             train=True, shuffle=self.shuffle)
        hash_stats_list = []
        self.model.eval()
        pbar = tqdm(total = len(train_data) * self.no_running_starts, smoothing = 0, position = 0)
        with torch.no_grad(): 
            for sample_no in range(self.no_running_starts):
                for batch in iter(running_starts_iter):
                    batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                    _, hash_stats, _, _ = self.model.get_rl_loss_for_batch(batch = batch,
                                                                          cost_manager = self.cost_manager,
                                                                          beta_entropy = self.beta_entropy,
                                                                          #loss_function = self.loss,
                                                                          use_cuda = self.use_cuda,
                                                                          max_output_length = self.max_output_length,
                                                                          level = self.level)
                    hash_stats_list.extend(hash_stats)
                    pbar.update(len(hash_stats))

        hash_stats_list*=self.running_starts_multiplier # will duplicate the list by this constant times -  1
        running_starts_avg_score, pct_failure = self.cost_manager.update_buffers(hash_stats_list)
        print(f"Average score during running starts was {running_starts_avg_score:.2f} and percent failure rate was {pct_failure:.2f}")
        self.cost_manager._save_trailing_stats()
        print(f"The trailing stats dict has been saved to {self.cost_manager.trailing_stats_out_path}")
        self.model.train()


    def _get_reference_baseline(self, train_data: Dataset, model: Model):
        reference_baseline_iter = make_data_iter(train_data,
                                             batch_size=self.running_starts_batch_size,
                                             batch_type=self.running_starts_batch_type,
                                             train=True, shuffle=self.shuffle)
        pbar = tqdm(total = len(train_data), smoothing = 0, position=0)
        for batch in iter(reference_baseline_iter):
            batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
            decoded_src = model.trg_vocab.arrays_to_sentences(arrays=batch.src, cut_at_eos=True)
            decoded_trg = model.trg_vocab.arrays_to_sentences(arrays = batch.trg, cut_at_eos=True)
            join_char = " " if self.level in ["word", "bpe"] else ""
            train_sources = [join_char.join(t) for t in decoded_src]
            train_references = [join_char.join(t) for t in decoded_trg]
            if self.level == "bpe":
                train_sources = [bpe_postprocess(s) for s in train_sources]
                train_references = [bpe_postprocess(s) for s in train_references]
            self.cost_manager.log_reference_baselines(zip(train_sources, train_references))
            pbar.update(len(train_sources))


    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """

        self.rl_adv_list = []
        #self._get_reference_baseline(train_data, self.model)
        if self.no_running_starts > 0:
            self._do_running_starts(train_data)

        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    batch_type=self.batch_type,
                                    train=True, shuffle=self.shuffle)

        # For last batch in epoch batch_multiplier needs to be adjusted
        # to fit the number of leftover training examples
        leftover_batch_size = len(
            train_data) % (self.batch_multiplier * self.batch_size)

        #self.current_batch_multiplier = self.batch_multiplier
        #self.count = self.current_batch_multiplier - 1
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            self.epoch_loss = 0
            self.current_batch_multiplier = self.batch_multiplier
            self.count = self.current_batch_multiplier - 1

            for i, batch in enumerate(iter(train_iter)):
                # reactivate training
                self.model.train()
                # create a Batch object from torchtext batch
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                if self.batch_type == "sentence":
                    if self.batch_multiplier > 1 and i == len(train_iter) - math.ceil(leftover_batch_size / self.batch_size):
                        self.current_batch_multiplier = math.ceil(leftover_batch_size / self.batch_size)
                        self.count = self.current_batch_multiplier - 1

                # calculate grads, accumulate, and step
                self._train_batch(batch)

                # log learning progress
                if self.steps % self.logging_freq == 0 and self.update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, self.multi_batch_score,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and self.update:
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_ppl, valid_sources, \
                        valid_sources_raw, valid_references, valid_hypotheses, \
                        valid_hypotheses_raw, valid_attention_scores, _ = \
                        validate_on_data(
                            logger=self.logger,
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            cost_manager = self.cost_manager,
                            level=self.level, model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            beam_size=1,  # greedy validations
                            batch_type=self.eval_batch_type
                        )

                    self.tb_writer.add_scalar("valid/valid_loss",
                                              valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score",
                                              valid_score, self.steps)
                    self.tb_writer.add_scalar("valid/valid_ppl",
                                              valid_ppl, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        ckpt_score = valid_ppl
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if self.scheduler is not None \
                            and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        valid_ppl=valid_ppl, eval_metric=self.eval_metric,
                        new_best=new_best)

                    self._log_examples(
                        sources_raw=[v for v in valid_sources_raw],
                        sources=valid_sources,
                        hypotheses_raw=valid_hypotheses_raw,
                        hypotheses=valid_hypotheses,
                        references=valid_references
                    )

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result (greedy) at epoch %3d, '
                        'step %8d: %s: %6.2f, loss: %8.4f, ppl: %8.4f, '
                        'duration: %.4fs', epoch_no + 1, self.steps,
                        self.eval_metric, valid_score, valid_loss,
                        valid_ppl, valid_duration)

                    # store validation set outputs
                    self._store_outputs(valid_hypotheses)

                    # store attention plots for selected valid sentences
                    #breakpoint()
                    if valid_attention_scores:
                        store_attention_plots(
                            attentions=valid_attention_scores,
                            targets=valid_hypotheses_raw,
                            sources=[s for s in valid_data.src],
                            indices=self.log_valid_sents,
                            output_prefix="{}/att.{}".format(
                                self.model_dir, self.steps),
                            tb_writer=self.tb_writer, steps=self.steps)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                    self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.2f',
                             epoch_no + 1, self.epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no + 1)
        self.logger.info('Best validation result (greedy) at step '
                         '%8d: %6.2f %s.', self.best_ckpt_iteration,
                         self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :param count: number of portions (batch_size) left before update
        :return: loss for batch (sum)
        """

        # the logic here will only update when count == 0, else we accumulate gradients

        hash_stats_list = []

        for sample_no in range(self.sentence_samples):

            # returns batch-loss calculated by a normalized advantage function
            # and a list of tuples of of hashes to a stats dictionary
            # we should cache the costs and then update the buffers for trailing costs after all samples are taken
            batch_loss, hash_stats, rl_adv, entropy = self.model.get_rl_loss_for_batch(batch = batch,
                                                                              cost_manager = self.cost_manager,
                                                                              beta_entropy = self.beta_entropy,
                                                                              use_cuda = self.use_cuda,
                                                                              max_output_length = self.max_output_length,
                                                                              level = self.level)

            self.rl_adv_list.extend(rl_adv)
            hash_stats_list.extend(hash_stats) # list of tuples (h, {"cost": , "tunit_fail":, "cost_fail"}

            if self.normalization == "batch":
                normalizer = batch.nseqs
            elif self.normalization == "tokens":
                normalizer = batch.ntokens
            elif self.normalization == "none":
                normalizer = 1
            else:
                raise NotImplementedError(
                    "Only normalize by 'batch' or 'tokens' "
                    "or summation of loss 'none' implemented")

            norm_batch_loss = batch_loss / normalizer
            entropy/=normalizer
            if self.current_batch_multiplier > 1:
                norm_batch_loss = norm_batch_loss/self.current_batch_multiplier \
                        if self.normalization != "none" else norm_batch_loss
                norm_batch_loss/=self.sentence_samples # normalize again for # samples taken
                self.multi_batch_loss += norm_batch_loss.detach() # accumulate loss for batch

                entropy = entropy/self.current_batch_multiplier \
                        if self.normalization != "none" else entropy
                entropy/=self.sentence_samples
                self.multi_batch_entropy += entropy # entropy is detached here

            norm_batch_loss.backward()
            del norm_batch_loss # explicitly deleting
            # end sampling loop

        # update buffers after sampling
        batch_score, pct_failures = self.cost_manager.update_buffers(hash_stats_list)
        self.multi_batch_hash_list.extend([h for h, _ in hash_stats_list])


        if self.current_batch_multiplier > 1:
            self.multi_batch_score += (batch_score / self.current_batch_multiplier \
                    if self.normalization != "none" else batch_score)
            self.multi_batch_pct_failure += pct_failures / self.current_batch_multiplier \
                    if self.normalization != "none" else pct_failures
            #print(f"multi_batch_score is {self.multi_batch_score} and batch_score is {batch_score} and multiplier is {self.current_batch_multiplier} and count is {self.count}")
        #if batch_score > 9999 or self.multi_batch_score > 9999: 
            #breakpoint()
        self.update = (self.count == 0)
        if self.update:
            if self.clip_grad_fun is not None:
                # clip gradients (in-place)
                self.clip_grad_fun(params=self.model.parameters())
            adv_std = np.std(self.rl_adv_list)
            adv_std = adv_std if adv_std != 0 else self.max_score
            for param in self.model.parameters():
                if param.grad != None:
                    param.grad.data/=adv_std

            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.rl_adv_list = []

            if self.scheduler is not None and \
                    self.scheduler_step_at == "step":
                self.scheduler.step()

            # increment step counter

            self.cost_manager.log_buffer_stats(self.multi_batch_hash_list)

            self.tb_writer.add_scalar("train/multi_batch_loss",
                                      self.multi_batch_loss, self.steps)
            self.tb_writer.add_scalar("train/multi_batch_score",
                                      self.multi_batch_score, self.steps)
            self.tb_writer.add_scalar("train/multi_batch_failure_rate",
                                      self.multi_batch_pct_failure,  self.steps)
            self.tb_writer.add_scalar("train/multi_batch_entropy",
                                      self.multi_batch_entropy, self.steps)
            self.tb_writer.add_scalar("train/no_baselines_beat",
                                      self.cost_manager.no_beat_baselines, self.steps)

            self.epoch_loss += self.multi_batch_loss
            self.log_batch_score += self.multi_batch_score

            self.multi_batch_hash_list = []
            self.multi_batch_loss = 0
            self.multi_batch_score = 0
            self.multi_batch_pct_failure = 0
            self.count = self.batch_multiplier - 1
            self.steps += 1

        else:
            self.count -= 1


    def _add_report(self, valid_score: float, valid_ppl: float,
                    valid_loss: float, eval_metric: str,
                    new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps, valid_loss, valid_ppl, eval_metric,
                    valid_score, current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        def f(p): 
            return p.requires_grad
        model_parameters = filter(f, self.model.parameters())
        #model_parameters = filter(lambda p: p.requires_grad,
        #                          self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(self, sources: List[str], hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            self.logger.info("Example #%d", p)

            if sources_raw is not None:
                self.logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                self.logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                self.logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            self.logger.info("\tSource:     %s", sources[p])
            self.logger.info("\tReference:  %s", references[p])
            self.logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.steps)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))

    def train_and_validate_actor_learner(self, valid_data: Dataset, src_field: Field, src_vocab, tgt_vocab):

        if self.shard_data:
            print("sharding the data")
            # shard_data(input_path: str, shard_path: str, src_lang: str, tgt_lang: str, n_shards: int)
            shard_data(input_path=self.train_path, shard_path = self.shard_path,
                       src_lang = self.src_lang, tgt_lang = self.tgt_lang, n_shards = self.n_actors)
        if self.use_shards:
            actor_data_prefixes = [self.shard_path + "_{}".format(i) for i in range(self.n_actors)]
        else:
            actor_data_prefixes = [self.train_path] * self.n_actors # use original data set
        #m = mp.Manager()
        global latest_model_id
        global model_lock
        latest_model_id = 0 # mp.Value("i", 0)
        #m = mp.Manager()
        #mp.set_start_method('spawn', force = True)
        #ctx = mp.get_context("spawn")
        model_lock = mp.Lock() #rwlock.RWLockWrite()
        self._save_learner()
        #trajectory_queue = mp.Queue(maxsize=self.replay_buffer_size)
        trajectory_queue = mp.Queue()
        generate_trajectory_flag = mp.Event()
        generate_trajectory_flag.set()
        running_starts_counter = mp.Value("i", 1) #self.n_actors)
        print(f"Learner has {torch.cuda.device_count()} available gpus and {torch.cuda.current_device()} is current device")

        device_indices = [i % len(self.actor_devices) for i in range(self.n_actors)]
        actor_device_list = [self.actor_devices[i] for i in device_indices]
        print(f"actor device list is {actor_device_list}")
        jobs = [{"model_cfg": self.model_cfg,
                "src_field": src_field,
                "hash2metadata": self.hash2metadata,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "path_to_data": path,
                "src_suffix": self.src_lang,
                "path_to_update_model": self.learner_model_path,
                "stoke_container_port_no": self.container_port,
                "generate_trajs_flag": generate_trajectory_flag,
                "latest_model_id": latest_model_id,
                "model_lock": model_lock,
                "trajs_queue": trajectory_queue,
                "running_starts_counter": running_starts_counter,
                "max_output_length": self.max_output_length,
                "level": self.level,
                "batch_size": self.batch_size,
                "pad_index": self.pad_index,
                "eos_index": self.eos_index, 
                "batch_type": self.batch_type,
                "device": device,
                "no_running_starts": self.no_running_starts,
                "actor_id": i,
                "performance_plot_path": "{}/actor_{}_performance_plot.png".format(self.model_dir, i)
                 } for i, (path, device) in enumerate(zip(actor_data_prefixes, actor_device_list))]
        #breakpoint()
        #actor_pool = mp.Pool(self.n_actors, initializer=init, initargs=(model_lock, latest_model_id, running_starts_counter))
        #actor_pool.map(actor_wrapper, jobs)
        processes = [mp.Process(target=actor_wrapper, args=(job,), daemon=True) for job in jobs]
        for p in processes: 
            p.start()
        ## TODO: verify the variables and variable names used here
        replay_buffer = BucketReplayBuffer(max_src_len=self.max_sent_length, max_output_len = self.max_output_length,
                                           n_splits = self.bucket_buffer_splits, max_buffer_size = self.replay_buffer_size)


        print(f"Main thread now executing {self.no_running_starts} on {self.n_actors} actors", flush = True)
        if self.no_running_starts > 0:
            while running_starts_counter.value > 0:
                replay_buffer.clear_queue(trajectory_queue, cost_manager=self.cost_manager)
                time.sleep(0.5)
        print("All running starts have finished", flush = True)
        print("Now checking if the replay buffer is filled.... if not, waiting for it to be filled", flush = True)

        while not replay_buffer.is_full():
            replay_buffer.clear_queue(queue=trajectory_queue, cost_manager=self.cost_manager)
            time.sleep(0.5)
        print("Replay buffer is filled, now training ", flush = True)

        if self.synchronized_al:
            train_output_fh = open(os.path.join(self.model_dir, "train_outputs.txt"), "w+")

        multi_batch_loss = 0
        multi_batch_entropy = 0
        multi_batch_n_seqs = 0
        multi_batch_n_tokens = 0
        multi_batch_advantage = 0
        multi_batch_costs = []
        multi_batch_failures = []

        performance_timer = StopWatch(name = "stopwatch")
        performance_timer.new_event("Model_Forward_Backward")
        performance_timer.new_event("Save_Learner")
        performance_timer.new_event("Clear_Queue")
        performance_timer.new_event("Validation_Testing")
        performance_timer.start()
        for step in range(1, self.n_updates * self.batch_multiplier):
            #print("starting and sampling from queue", flush = True)
            #breakpoint()
            performance_timer.Model_Forward_Backward.start()
            self.model.train()
            if self.synchronized_al:
                src_inputs, traj_outputs, log_probs, advantages, costs, corrects, failed, src_lens, tgt_lens, \
                    result_strings = replay_buffer.synchronous_sample(
                    queue=trajectory_queue,max_size=self.batch_size, cost_manager=self.cost_manager, step_no=step/self.batch_multiplier)
                if ((step/self.batch_multiplier) % 10) == 0:
                    train_output_fh.write("\n\n".join(result_strings))
                multi_batch_costs.extend(costs)
                multi_batch_failures.extend(failed)
            else:
                src_inputs, traj_outputs, log_probs, advantages, costs, corrects, failed, src_lens, tgt_lens = replay_buffer.sample(max_size = self.batch_size, cost_manager=self.cost_manager)

            #print("queue samples, now processing batch", flush = True)
            batch = LearnerBatch(src_seqs = src_inputs, tgt_seqs = traj_outputs, log_probs=log_probs, advantages=advantages,
                                 pad_index = self.pad_index, bos_index = self.bos_index)
            batch.to_device(self.learner_device)
            #print("batch processed and on device, doing forward", flush = True)
            out, hidden, att_probs, _ = self.model.forward(src=batch.src,
                                                           trg_input=batch.tgt_input,
                                                           src_mask=batch.src_mask,
                                                           src_lengths=src_lens,
                                                           trg_mask=batch.tgt_mask,
                                                           )
            online_log_probs, online_entropy = log_probs_and_entropy(logits = out, labels = batch.tgt, loss_mask = batch.loss_mask)
            offline_log_probs = batch.offline_log_probs * batch.loss_mask
            online_traj_probs = torch.sum(online_log_probs.detach(), dim = 1).unsqueeze(1) # should reduce to a 2-d array, but with dim 1 = 1
            offline_traj_probs = torch.sum(offline_log_probs, dim = 1).unsqueeze(1) # should reduce to a 2-d array
            if self.ppo_flag:
                importance_sampling_ratio = torch.exp(online_traj_probs - offline_traj_probs)
                clipped_importance_sampling_ratio = torch.clamp(importance_sampling_ratio,
                                                               min=1-self.ppo_epsilon, max = 1+self.ppo_epsilon)
                advantages = torch.max(batch.advantages*importance_sampling_ratio,
                                       batch.advantages*clipped_importance_sampling_ratio)
            else:
                advantages = batch.advantages
            #advantages = advantages - torch.mean(advantages)
            #advantages/=torch.max(torch.std(advantages), torch.ones_like(advantages))
            #advantages[advantages>0] = 0
            n_tokens = sum(tgt_lens)
            # maximizing exploration entropy = minimizing negative entropy
            loss = torch.sum(online_log_probs * advantages) - online_entropy * self.beta_entropy
            loss /= n_tokens# normalize by the number of total output tokens
            loss /= self.batch_multiplier # normalize by the batch multiplier
            #print("loss processed, now backward", flush = True)
            loss.backward()
            multi_batch_loss += loss.detach().cpu().item()
            multi_batch_entropy += (online_entropy.detach().cpu().item() / n_tokens)/self.batch_multiplier
            multi_batch_n_seqs += len(batch.src)
            multi_batch_n_tokens += n_tokens
            multi_batch_advantage += torch.mean(advantages).item()/self.batch_multiplier

            #print("backward done", flush = True)
            performance_timer.Model_Forward_Backward.stop()

            if (step % self.batch_multiplier) == 0:
                update_no = step // self.batch_multiplier
                #print("optimizer step", flush = True)
                if self.clip_grad_fun is not None:
                    # clip gradients (in-place)
                    self.clip_grad_fun(params=self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()
                performance_timer.Save_Learner.start()
                #print("step done, saving learner", flush = True)
                if (update_no % self.save_learner_every) == 0: 
                    self._save_learner()
                performance_timer.Save_Learner.stop()

                if not self.synchronized_al:
                    performance_timer.Clear_Queue.start()
                    #print("saving done, clearing queue", flush = True)
                    avg_queue_cost, avg_queue_failures, new_examples = replay_buffer.clear_queue(trajectory_queue, cost_manager = self.cost_manager)
                    performance_timer.Clear_Queue.stop()
                    new_examples = max(new_examples, 1)
                    self.tb_writer.add_scalar("train/number-trained-per-new-observations",
                                              multi_batch_n_seqs / new_examples, update_no)
                    self.tb_writer.add_scalar("train/queue_size",
                                              new_examples, update_no)
                else:
                    avg_queue_cost = np.mean(multi_batch_costs)
                    avg_queue_failures = np.mean(multi_batch_failures)
                #print("queue done, tensorboard writing", flush = True)

                if avg_queue_cost > 0: 
                    self.tb_writer.add_scalar("train/avg_cost",
                                              avg_queue_cost, update_no)
                    self.tb_writer.add_scalar("train/avg_failure_rate",
                                              avg_queue_failures, update_no)

                self.tb_writer.add_scalar("train/batch_size",
                                          multi_batch_n_seqs, update_no)
                self.tb_writer.add_scalar("train/no_baselines_beat",
                                          self.cost_manager.no_beat_baselines, update_no)
                self.tb_writer.add_scalar("train/avg_entropy",
                                          multi_batch_entropy, update_no)
                self.tb_writer.add_scalar("train/batch_size_tokens",
                                          multi_batch_n_tokens, update_no)
                self.tb_writer.add_scalar("train/batch_loss", 
                                            multi_batch_loss, update_no)
                self.tb_writer.add_scalar("train/avg_advantage", 
                                                multi_batch_advantage, update_no)

                multi_batch_loss = 0
                multi_batch_entropy = 0
                multi_batch_n_seqs = 0
                multi_batch_n_tokens = 0
                multi_batch_advantage = 0
                multi_batch_costs = []
                multi_batch_failures = []
                #print("tensorboard writing done, update no {}".format(update_no), flush = True)
                if (update_no % 5000) == 0: 
                    state = {"model_state": self.model.state_dict()}
                    torch.save(state, "{}/model_{}.ckpt".format(self.model_dir, update_no)) 
                if (update_no % self.log_best_seq_stats_every) == 0:
                    #print("inside cost manager save best seq stats", flush = True)
                    #pdb.set_trace()
                    self.cost_manager.save_best_seq_stats()
                    #print("saved those stats", flush = True)

                #print(f"update no is {update_no} and valdation_freq is {self.validation_freq}")
                #print(f"eval modulo evaluates as {(update_no % self.validation_freq)}")
                if (update_no % self.validation_freq) == 0:
                #if update_no > 300: 
                    print("doing validation loop")
                    performance_timer.Validation_Testing.start()
                    self.model.eval()
                    valid_score, valid_loss, valid_ppl, valid_sources, \
                    valid_sources_raw, valid_references, valid_hypotheses, \
                    valid_hypotheses_raw, valid_attention_scores, _ = \
                        validate_on_data(
                            logger=self.logger,
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            cost_manager=self.cost_manager,
                            level=self.level, model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            beam_size=1,  # greedy validations
                            batch_type=self.eval_batch_type
                        )

                    self.tb_writer.add_scalar("valid/valid_cost",
                                              valid_score, update_no)

                    ckpt_score = valid_score

                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            self._save_checkpoint()
                    performance_timer.Validation_Testing.stop()

        self.cost_manager._save_trailing_stats()
        shutil.copy2(self.cost_manager.trailing_stats_out_path, os.path.join(self.model_dir, "ending_trailing_stats.pkl"))
        # gracefully shut down
        performance_timer.stop()
        print("shutting down the child processes")
        generate_trajectory_flag.clear()
        for p in processes:
            p.join(timeout=10)
        train_output_fh.close()
        print("making performance plot")
        performance_timer.make_perf_plot(title = "Learner Performance Benchmarking",
                                         path = "{}/learner_perf_plot.png".format(self.model_dir))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab, src_field, trg_field = load_data(
        data_cfg=cfg["data"])

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(train_data=train_data, valid_data=dev_data,
                  test_data=test_data, src_vocab=src_vocab, trg_vocab=trg_vocab,
                  logging_function=trainer.logger.info)

    trainer.logger.info(str(model))

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    actor_learner_flag = cfg["training"].get("actor_learner", False)
    if actor_learner_flag:
        trainer.train_and_validate_actor_learner(valid_data=dev_data, src_field=src_field,
                                                 src_vocab=src_vocab, tgt_vocab=trg_vocab)
    else:
        trainer.train_and_validate(train_data=train_data, valid_data=dev_data, src_vocab=src_vocab, tgt_vocab=trg_vocab)

    # predict with the best model on validation and test
    # (if test data is available)
    #ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    #output_name = "{:08d}.hyps".format(trainer.best_ckpt_iteration)
    #output_path = os.path.join(trainer.model_dir, output_name)
    #test(cfg_file, ckpt=ckpt, output_path=output_path, logger=trainer.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
