import torch
import numpy as np
import multiprocessing as mp
from helpers import load_checkpoint, hash_file
from modeling import Model
from vocabulary import Vocabulary
from torchtext.data import Field
from data import MonoDataset, make_data_iter
from batch import Batch
from helpers import bpe_postprocess, bpe2formatted, cut_arrays_at_eos, slice_arrays_with_lengths, is_unique_batch
from typing import Dict
from req import StokeRequest
from prwlock import RWLock
from fairseq import pdb

def deduplicate_tensor(data: torch.Tensor):
    arr = np.unique(data.numpy(), axis=0)
    return torch.from_numpy(arr)

def get_trajs(model: Model, batch: Batch, max_output_length: int, level: str, eos_index: int, device: str):
    # sort batch now by src length and keep track of order
    #batch_tensor = deduplicate_tensor(batch_tensor)
    # TODO : find another way to deduplicate

    sort_reverse_index = batch.sort_by_src_lengths()

    # run as during inference to produce translations & RL score
    with torch.no_grad():
        output, transposed_log_probs, entropy = model.run_rl_batch(
            batch=batch, max_output_length=max_output_length)

    # sort outputs back to original order
        output = output[sort_reverse_index]
        log_probs = torch.stack(transposed_log_probs).T[sort_reverse_index]  # T x B -> B x T as Tensor



    # decode back to symbols
    decoded_src = model.src_vocab.arrays_to_sentences(arrays=batch.src,
                                                     cut_at_eos=True)
    decoded_hyp = model.trg_vocab.arrays_to_sentences(arrays=output,
                                                     cut_at_eos=True)

    # evaluate with metric on full dataset
    join_char = " " if level in ["word", "bpe"] else ""
    src_bpe_strings = [join_char.join(s) for s in decoded_src]
    hypothesis_bpe_strings = [join_char.join(t) for t in decoded_hyp]

    # post-process
    if level == "bpe":
        src_bpe_strings = [bpe_postprocess(s) for s in src_bpe_strings]
        hypothesis_bpe_strings = [bpe_postprocess(v) for
                            v in hypothesis_bpe_strings]

    # clean up the outputs and log probs by stripping any post-eos as well as padding
    output_list, output_lengths = cut_arrays_at_eos(list(output.cpu().numpy()), eos_index = eos_index)
    log_probs_list = slice_arrays_with_lengths(list(log_probs.cpu().numpy()), output_lengths)

    return output_list, log_probs_list, list(entropy.cpu()), output_lengths, src_bpe_strings, hypothesis_bpe_strings


def eval_trajs(source_bpe_stings: str, hypothesis_bpe_strings: str, hash2metadata: Dict, requester: StokeRequest):
    jobs = {}
    formatted_hyps = []
    for source_bpe_str, hypothesis_bpe_str in zip(source_bpe_stings, hypothesis_bpe_strings):
        h = hash_file(source_bpe_str.strip())
        if h in jobs:
            print(f"duplicate for {hash2metadata[h]['name']}")
        metadata = hash2metadata[h]
        formatted_hypothesis, _ = bpe2formatted(assembly_string = hypothesis_bpe_str, function_name = metadata["name"],
                                                remove_header = True, remove_footer = True)
        jobs[h] = {"hypothesis_string": formatted_hypothesis, "metadata": metadata}
        formatted_hyps.append(formatted_hypothesis)
    results = requester.get(jobs)
    hashes = []
    stats_list = []
    metadatas = []

    for h, result_dict in results.items():
        hashes.append(h)

        stats_list.append(result_dict["stats"])
        metadatas.append(result_dict["metadata"])

    return hashes, stats_list, metadatas, formatted_hyps


def prune_hash2metadata(hash2metadata: Dict, model: Model, level: str, data_iter):
    new_hash2metadata = {}
    for batch in iter(data_iter):
        decoded_src = model.src_vocab.arrays_to_sentences(arrays=batch.src,
                                                          cut_at_eos=True)
        join_char = " " if level in ["word", "bpe"] else ""
        sources = [join_char.join(s) for s in decoded_src]
        if level == "bpe":
            sources = [bpe_postprocess(s) for s in sources]
        for s in sources:
            h = hash_file(s.strip())
            new_hash2metadata[h] = hash2metadata[h]
    return new_hash2metadata


def actor(model: Model, src_field: Field, hash2metadata: Dict,
          path_to_data: str, src_suffix: str,
          path_to_update_model: str, stoke_container_port_no: str,
          generate_trajs_flag: mp.Event, latest_model_id: mp.Value, model_lock: RWLock, trajs_queue: mp.Queue,
          running_starts_counter: mp.Value, max_output_length: int, level: str, batch_size: int, pad_index: int,
          no_running_starts: int, actor_id: int, batch_type: str = "token", device: str = "cpu") -> None:
    print(f"actor id is {actor_id}", flush = True)
    if actor_id == 0: 
        pdb.set_trace()
    current_model_id = latest_model_id.value
    with model_lock:
        model_checkpoint = load_checkpoint(path = path_to_update_model, use_cuda = False)
    model.load_state_dict(model_checkpoint["model_state"])
    model.to(device)

    data = MonoDataset(path=path_to_data, ext="." + src_suffix,
                            field=src_field)

    data_iter = make_data_iter(data,
                                batch_size=batch_size,
                                batch_type=batch_type,
                                train=True, shuffle=True)

    hash2metadata = prune_hash2metadata(hash2metadata=hash2metadata, model=model, level=level, data_iter=data_iter)
    requester = StokeRequest(port = stoke_container_port_no)

    running_starts_left = no_running_starts
    running_starts_flag = True if running_starts_left > 0 else False

    while generate_trajs_flag.is_set():
        for batch in iter(data_iter):

            # ensure no batch duplicates
            if not is_unique_batch(batch):
                continue

            batch = Batch(batch, pad_index=pad_index, use_cuda=False)
            src, src_lengths, n_seqs = batch.src, batch.src_lengths, batch.neqs
            src_list = slice_arrays_with_lengths(list(src.cpu().numpy()), list(src_lengths.numpy()))
            batch.to_device(device)

            with model_lock.reader_lock():
                tmp_model_id = latest_model_id.value()
                if current_model_id != tmp_model_id:
                    current_model_id = tmp_model_id
                    model_checkpoint = load_checkpoint(path=path_to_update_model, use_cuda=False)
                    model.load_state_dict(model_checkpoint["model_state"])
                    model.to(device)

            #output_list, log_probs_list, list(entropy.cpu()), output_lengths, src_bpe_strings, hypothesis_bpe_strings
            output_list, log_prob_list, entropy_list, output_lengths, src_bpe_strings, hypothesis_bpe_strings = \
                get_trajs(model=model,
                    batch_tensor=batch,
                    max_output_length=max_output_length,
                    level=level,
                    pad_index=pad_index,
                    device=device)

            #TODO update the local metadata dictionary based off the api output

            hashes, stats_list, metadatas, formatted_hyps = eval_trajs(source_bpe_stings=src_bpe_strings,
                                                                         hypothesis_bpe_strings=hypothesis_bpe_strings,
                                                                         hash2metadata=hash2metadata,
                                                                         requester=requester)
            for h, metadata_update in zip(hashes, metadatas):
                hash2metadata[h] = metadata_update

            for hash, src_input, traj_output, log_probs, stats, formatted_hyp, src_len, out_len in \
               zip(hashes, src_list, output_list, log_prob_list, stats_list, formatted_hyps,
                   src_lengths, output_lengths):

                trajs_queue.put({"hash": hashes,
                                    "src_input": src_input,
                                    "traj_output": traj_output,
                                    "log_probs": log_probs,
                                    "stats": stats,
                                    "formatted_hyp": formatted_hyp,
                                    "src_len": src_len,
                                    "out_len": out_len
                                    })

            if running_starts_flag:
                running_starts_left-=1
                print(f"actor number {actor_id} has {running_starts_left} running starts left", flush=True)
                if running_starts_left == 0:
                    running_starts_flag = False
                    with running_starts_counter.get_lock():
                        running_starts_counter.value -= 1
                    print(f"... and actor number {actor_id} has finished running starts", flush=True)


            if not generate_trajs_flag.is_set():
                break







