# coding: utf-8

"""
Implementation of a mini-batch.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Union

class LearnerBatch:
    def __init__(self, src_seqs: List[Union[np.array, torch.Tensor]],
                 tgt_seqs: List[Union[np.array, torch.Tensor]],
                 log_probs: List[Union[np.array, torch.Tensor]],
                 advantages: List[int], pad_index: int, bos_index: int):
        if type(src_seqs[0]) != torch.Tensor:
            src_seqs = [torch.tensor(s) for s in src_seqs]
        if type(tgt_seqs[0]) != torch.Tensor:
            tgt_seqs = [torch.tensor(t) for t in tgt_seqs]
        if type(log_probs[0]) != torch.Tensor:
            log_probs = [torch.tensor(lp) for lp in log_probs]
        src_seqs = [torch.tensor(src) for src in src_seqs]
        tgt_seqs = [torch.tensor(tgt) for tgt in tgt_seqs]
        self.src = pad_sequence(src_seqs, batch_first=True, padding_value=pad_index)
        self.src_mask = (self.src != pad_index).unsqueeze(1) # per Batch outlined below
        tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_index)

        self.tgt_input = tgt[:, :-1] # like in Batch, the index is BOS -> second to last token to allow last token to be predicted
        self.tgt = tgt[:, 1:] # likewise, the tgt is offset by 1
        self.tgt_mask = (self.tgt_input != pad_index).unsqueeze(1) # for use with transformers, not for RNNs

        self.loss_mask = (self.tgt != pad_index) # do not unsqueeze as this is used for post-processing 
        # to match self.tgt we do not need to slice, as we do not add any additional <bos>
        self.offline_log_probs = pad_sequence(log_probs, batch_first=True, padding_value=0.0)
        #self.loss_mask = (self.offline_log_probs !=0.0) # do not unsqueeze as this is used for post-processing 

        self.advantages = torch.tensor(advantages).unsqueeze(1).type(torch.float) # turn into a tensor, but 1D -> 1D with extra 1 dimension for broadcasting

    def to_device(self, device: str):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_input = self.tgt_input.to(device)
        self.tgt = self.tgt.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.loss_mask = self.loss_mask.to(device)
        self.offline_log_probs = self.offline_log_probs.to(device)
        self.advantages = self.advantages.to(device)


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def to_device(self, device: str):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)


    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index
