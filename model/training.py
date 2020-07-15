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

# import math
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
from modeling import build_model
from batch import Batch
from helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError
from modeling import Model
from prediction import validate_on_data
from loss import XentLoss, StokeCostManager
from data import load_data, make_data_iter
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from prediction import test
import gc

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
        self.cost_manager = StokeCostManager(hash2metadata = self.hash2metadata,
                                             container_name = data_config.get("container_name"),
                                             host_path_to_volume = data_config.get("host_path_to_volume"),
                                             container_path_to_volume = data_config.get("container_path_to_volume"),
                                             volume_path_to_data = data_config.get("volume_path_to_data"),
                                             volume_path_to_tmp = data_config.get("volume_path_to_tmp"),
                                             tb_writer = self.tb_writer,
                                             n_best_seq_dir="{}/best_seqs/".format(self.model_dir),
                                             max_len = data_config.get("max_len"),
                                             max_score = data_config.get("max_score"),
                                             n_workers = data_config.get("n_workers"),
                                             keep_n_best_seqs=data_config.get("keep_n_best_seqs", 10),
                                             )


        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()

        # objective
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
        if self.early_stopping_metric in ["ppl", "loss"]:
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
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(model_load_path,
                                      reset_best_ckpt=reset_best_ckpt,
                                      reset_scheduler=reset_scheduler,
                                      reset_optimizer=reset_optimizer)

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
        with torch.no_grad(): 
            for sample_no in range(self.no_running_starts):
                for batch in iter(running_starts_iter):
                    batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                    _, hash_stats = self.model.get_rl_loss_for_batch(batch = batch,
                                                                          cost_manager = self.cost_manager,
                                                                          loss_function = self.loss,
                                                                          use_cuda = self.use_cuda,
                                                                          max_output_length = self.max_output_length,
                                                                          level = self.level)
                    hash_stats_list.extend(hash_stats)

        hash_stats_list*=self.running_starts_multiplier # will duplicate the list by this constant times -  1
        running_starts_avg_score = self.cost_manager.update_buffers(hash_stats_list)
        print(f"Average score during running starts was {running_starts_avg_score:.2f}")
        self.model.train()

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        breakpoint()
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

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            self.current_batch_multiplier = self.batch_multiplier
            self.count = self.current_batch_multiplier - 1
            self.epoch_loss = 0

            for i, batch in enumerate(iter(train_iter)):
                # reactivate training
                self.model.train()
                # create a Batch object from torchtext batch
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)

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
                        valid_hypotheses_raw, valid_attention_scores = \
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
            batch_loss, hash_stats = self.model.get_rl_loss_for_batch(batch = batch,
                                                                      cost_manager = self.cost_manager,
                                                                      loss_function = self.loss,
                                                                      use_cuda = self.use_cuda,
                                                                      max_output_length = self.max_output_length,
                                                                      level = self.level)

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
            if self.current_batch_multiplier > 1:
                norm_batch_loss = norm_batch_loss/self.current_batch_multiplier \
                        if self.normalization != "none" else norm_batch_loss
                norm_batch_loss/=self.sentence_samples # normalize again for # samples taken
                self.multi_batch_loss += norm_batch_loss.detach() # accumulate loss for batch

            norm_batch_loss.backward()
            del norm_batch_loss # explicitly deleting
            # end sampling loop

        # update buffers after sampling
        batch_score = self.cost_manager.update_buffers(hash_stats_list)

        if self.current_batch_multiplier > 1:
            self.multi_batch_score += (batch_score / self.current_batch_multiplier \
                    if self.normalization != "none" else batch_score)

        self.update = (self.count == 0)
        if self.update:
            if self.clip_grad_fun is not None:
                # clip gradients (in-place)
                self.clip_grad_fun(params=self.model.parameters())

            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None and \
                    self.scheduler_step_at == "step":
                self.scheduler.step()

            # increment step counter

            self.cost_manager.log_buffer_stats()

            self.tb_writer.add_scalar("train/multi_batch_loss",
                                      self.multi_batch_loss, self.steps)
            self.tb_writer.add_scalar("train/multi_batch_score",
                                      self.multi_batch_score, self.steps)

            self.epoch_loss += self.multi_batch_loss
            self.log_batch_score += self.multi_batch_score

            self.multi_batch_loss = 0
            self.multi_batch_score = 0
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
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
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


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
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
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "{:08d}.hyps".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=trainer.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
