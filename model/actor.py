import torch
import numpy as np
import multiprocessing as mp
from helpers import load_checkpoint, hash_file
from modeling import Model, build_model
from vocabulary import Vocabulary
from torchtext.data import Field
from data import MonoDataset, make_data_iter
from batch import Batch
from helpers import bpe_postprocess, bpe2formatted, cut_arrays_at_eos, slice_arrays_with_lengths, is_unique_batch, \
    StopWatch
from typing import Dict, List
from req import StokeRequest
from prwlock import RWLock
from fairseq import pdb


def deduplicate_tensor(data: torch.Tensor):
    arr = np.unique(data.numpy(), axis=0)
    return torch.from_numpy(arr)


def get_trajs(model: Model, batch: Batch, max_output_length: int, level: str, bos_index: int, eos_index: int,
              src_list: List, src_lengths: List):

    # sort batch now by src length and keep track of order
    sort_reverse_index = batch.sort_by_src_lengths()
    # run as during inference to produce translations & RL score
    with torch.no_grad():
        output, transposed_log_probs, entropy = model.run_rl_batch(
            batch=batch, max_output_length=max_output_length)
        # sort outputs back to original order
        output = output[sort_reverse_index]
        log_probs = torch.stack(transposed_log_probs).T[sort_reverse_index]  # T x B -> B x T as Tensor

        # decode back to symbols
        decoded_src = model.src_vocab.arrays_to_sentences(arrays=batch.src, cut_at_eos=True)
        decoded_hyp = model.trg_vocab.arrays_to_sentences(arrays=output, cut_at_eos=True)

        # then pre-pend the <BOS> to the outputs such that they can be processed by the learner
        # TODO: do in numpy to be more simple
        output = torch.from_numpy(output)
        bos_vec = output.new_full(size=[output.size(0), 1], fill_value=bos_index, dtype = torch.long)
        output = torch.cat((bos_vec, output), dim = 1).numpy() # add the bos along T dimension

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
    output_list, output_lengths = cut_arrays_at_eos(list(output), eos_index = eos_index)
    log_probs_list = slice_arrays_with_lengths(list(log_probs.cpu().numpy()), output_lengths)

    trajs_dict = {}
    for i, (src_bpe_string, hyp_bpe_string, traj_output, log_probs, src_input, src_len, out_len) in \
        enumerate(zip(src_bpe_strings, hypothesis_bpe_strings, output_list,
                      log_probs_list, src_list, src_lengths, output_lengths)):
        h = hash_file(src_bpe_string.strip())
        trajs_dict[str(i)] = {"hash": h, "hyp_bpe_string": hyp_bpe_string, "src_bpe_string": src_bpe_string,
    "traj_output": traj_output,"log_probs": log_probs, "src_input": src_input, "src_len": src_len, "out_len": out_len,
                         }
    return trajs_dict


def eval_trajs(trajs_dict: Dict, hash2metadata: Dict, requester: StokeRequest):
    jobs = {}

    for i, traj_dict in trajs_dict.items():
        h = traj_dict["hash"]
        metadata = hash2metadata[h]
        assert metadata["hash"] == h
        hypothesis_bpe_str = traj_dict["hyp_bpe_string"]
        src_bpe_str = traj_dict["src_bpe_string"]
        formatted_hypothesis, _ = bpe2formatted(assembly_string=hypothesis_bpe_str, function_name=metadata["name"],
                                                remove_header=True, remove_footer=True)
        formatted_src, _ = bpe2formatted(assembly_string=src_bpe_str, function_name=metadata["name"],
                                                remove_header=True, remove_footer=True)
        jobs[i] = {"hypothesis_string": formatted_hypothesis, "metadata": metadata}
        trajs_dict[i]["formatted_hyp"] = formatted_hypothesis
        trajs_dict[i]["formatted_src"] = formatted_src

    results = requester.get(jobs)
    update_hash2metadata = {}
    for i in trajs_dict.keys():
        h = trajs_dict[i]["hash"]
        result_dict = results[i]
        assert result_dict["metadata"]["hash"] == h
        trajs_dict[i]["stats"] = result_dict["stats"]
        update_hash2metadata[h] = result_dict["metadata"]

    return trajs_dict, update_hash2metadata


def prune_hash2metadata(hash2metadata: Dict, model: Model, level: str, data_iter, pad_index):
    new_hash2metadata = {}
    for batch in iter(data_iter):
        batch = Batch(batch, pad_index, use_cuda=False)
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


def actor_wrapper(args_dict): 
    return actor(**args_dict)


def actor(model_cfg: Dict, src_field: Field, hash2metadata: Dict, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
          path_to_data: str, src_suffix: str,
          path_to_update_model: str, stoke_container_port_no: str,
          generate_trajs_flag: mp.Event, latest_model_id: mp.Value,  model_lock: RWLock, running_starts_counter: mp.Value,
          trajs_queue: mp.Queue, max_output_length: int, level: str, batch_size: int, pad_index: int, eos_index: int, 
          bos_index: int, no_running_starts: int, actor_id: int, performance_plot_path: str,
          batch_type: str = "token", device: str = "cpu") -> None:
    batch_size/=2    #2
    print(f"actor id is {actor_id}", flush = True)
    #if actor_id == 0: 
        #pdb.set_trace()
    model = build_model(model_cfg, src_vocab=src_vocab, trg_vocab=tgt_vocab)
    current_model_id = latest_model_id #.value
    #with model_lock.reader_lock():
    with model_lock: 
        model_checkpoint = load_checkpoint(path = path_to_update_model, use_cuda = False)
    model.load_state_dict(model_checkpoint["model_state"], strict=False)
    #model.eval()
    #for model_key, ckpt_key in zip(model.state_dict(), model_checkpoint["model_state"]): 
    #    model_param = model.state_dict()[model_key]
    #    ckpt_param = model_checkpoint["model_state"][ckpt_key]
    #    model_param.data = ckpt_param.data

    print(f"desired device is {device} and trying to put on, {torch.cuda.device_count()} counts and {torch.cuda.current_device()} is current device")
    model.to(device)
    print(f"desired device is {device} and it succeeded !")
    data = MonoDataset(path=path_to_data, ext="." + src_suffix,
                            field=src_field)

    data_iter = make_data_iter(data,
                                batch_size=batch_size,
                                batch_type=batch_type,
                                train=True, shuffle=True)
    hash2metadata = prune_hash2metadata(hash2metadata=hash2metadata, model=model, level=level, data_iter=data_iter, pad_index = pad_index)
    requester = StokeRequest(base_url = "http://127.0.0.1", port = stoke_container_port_no)

    running_starts_left = no_running_starts
    running_starts_flag = True if running_starts_left > 0 else False

    performance_timer = StopWatch(name="stopwatch")
    performance_timer.new_event("Process_Batch")
    performance_timer.new_event("Load_Model")
    performance_timer.new_event("Sample_New_Rewrite")
    performance_timer.new_event("Evaluate_With_Stoke")
    performance_timer.new_event("Add_to_Queue")
    performance_timer.start()
    while generate_trajs_flag.is_set(): 
        for batch in iter(data_iter):

            # ensure no batch duplicates
            if not is_unique_batch(batch):
                continue
            #pdb.set_trace()
            performance_timer.Process_Batch.start()
            batch = Batch(batch, pad_index=pad_index, use_cuda=False)
            src, src_lengths, n_seqs = batch.src, batch.src_lengths, batch.nseqs
            src_list = slice_arrays_with_lengths(list(src.cpu().numpy()), list(src_lengths.numpy()))
            batch.to_device(device)
            performance_timer.Process_Batch.stop()
            performance_timer.Load_Model.start()
            with model_lock: 
                model_checkpoint = load_checkpoint(path=path_to_update_model, use_cuda=False)
            model.load_state_dict(model_checkpoint["model_state"])
            model.to(device)
            performance_timer.Load_Model.stop()
            performance_timer.Sample_New_Rewrite.start()
            trajs_dict = get_trajs(model=model,batch=batch, max_output_length=max_output_length, level=level,
                                   bos_index=bos_index, eos_index=eos_index, src_list=src_list, src_lengths=src_lengths)
            performance_timer.Sample_New_Rewrite.stop()
            performance_timer.Evaluate_With_Stoke.start()
            #def eval_trajs(trajs_dict: Dict, hash2metadata: Dict, requester: StokeRequest):
            trajs_dict, update_hash2metadata = eval_trajs(trajs_dict=trajs_dict, hash2metadata=hash2metadata,
                                                          requester=requester)
            for h, metadata_update in update_hash2metadata.items():
                hash2metadata[h] = metadata_update
            performance_timer.Evaluate_With_Stoke.stop()
            #pdb.set_trace()
            performance_timer.Add_to_Queue.start()
            for h, traj_dict in trajs_dict.items():
                trajs_queue.put({ **traj_dict})
            performance_timer.Add_to_Queue.stop()
            #pdb.set_trace()
            if running_starts_flag:
                running_starts_left-=1
                print(f"actor number {actor_id} has {running_starts_left} running starts left", flush=True)
                if running_starts_left == 0:
                    running_starts_flag = False
                    with running_starts_counter.get_lock():
                        running_starts_counter.value -= 1
                    print(f"... and actor number {actor_id} has finished running starts", flush=True)


            if not generate_trajs_flag.is_set():
                if performance_timer.timing: 
                    performance_timer.stop()
                performance_timer.make_perf_plot(title="Actor no {} Performance Benchmarking".format(actor_id),
                                                 path=performance_plot_path)
                break

