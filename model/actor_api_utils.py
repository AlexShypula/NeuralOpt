from vocabulary import Vocabulary

from helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, bpe_postprocess, BucketReplayBuffer

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from torchtext import data
from data import shard_data

import json


def char_tok(s):
    return list(s)

def word_tok(s):
    return s.split()


def parse_config(cfg_path: str):
    cfg = load_config(cfg_path)
    train_config = cfg["training"]
    data_config = cfg["data"]
    model_config = cfg["model"]

    level = data_config["level"]
    lowercase = data_config["lowercase"]
    src_vocab_file = data_config["src_vocab"]
    tgt_vocab_file = data_config["trg_vocab"]

    src_vocab = Vocabulary(file=src_vocab_file)
    tgt_vocab = Vocabulary(file=tgt_vocab_file)

    if level == "char":
        tok_fun = char_tok
    else:
        tok_fun = word_tok

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    src_field.vocab = src_vocab
    trg_field.vocab = tgt_vocab

    batch_size = train_config["batch_size"]
    batch_type = train_config.get("batch_type", "sentence")

    n_actors = train_config.get("n_actors", 1)
    actor_devices = train_config.get("actor_devices", "cpu").split(":")
    # actor-learner required data
    shard_data = data_config.get("shard_data", True)
    shard_path = data_config.get("shard_path", None)
    if shard_data:
        assert shard_path, "if sharding the data, a shard path must be specified"
    src_lang = data_config["src"]
    tgt_lang = data_config["trg"]
    train_path = data_config["train"]
    # self.dev_path = data_config["dev"]
    # self.test_path = data_config.get("test", None)
    max_sent_length = data_config["max_sent_length"]
    stoke_container_port = data_config.get("container_port", 6000)
    model_dir = train_config["model_dir"]
    learner_model_path = "{}/learner.ckpt".format(model_dir)

    bos_index = tgt_vocab.stoi[BOS_TOKEN]
    pad_index = tgt_vocab.stoi[PAD_TOKEN]
    eos_index = tgt_vocab.stoi[EOS_TOKEN]

    max_output_length = train_config.get("max_output_length", None)
    no_running_starts = train_config.get("no_running_starts", 0)

    with open(data_config.get("hash2metadata")) as fh:
        hash2metadata = json.load(fh)

    if shard_data:
        # shard_data(input_path: str, shard_path: str, src_lang: str, tgt_lang: str, n_shards: int)
        shard_data(input_path=train_path, shard_path=shard_path,
                   src_lang=src_lang, tgt_lang=tgt_lang, n_shards=n_actors)

    actor_data_prefixes = [shard_path + "_{}".format(i) for i in range(n_actors)]

    device_indices = [i % len(actor_devices) for i in range(n_actors)]
    actor_device_list = [actor_devices[i] for i in device_indices]


    parsed_config = {"model": model_config,
     "src_field": src_field,
     "hash2metadata": hash2metadata,
     "src_vocab": src_vocab,
     "tgt_vocab": tgt_vocab,
     #"path_to_data": path,
     "src_suffix": src_lang,
     "path_to_update_model": learner_model_path,
     "stoke_container_port_no": stoke_container_port,
     #"generate_trajs_flag": generate_trajectory_flag,
     #"latest_model_id": latest_model_id,
     #"model_lock": model_lock,
     #"trajs_queue": trajectory_queue,
     #"running_starts_counter": running_starts_counter,
     "max_output_length": max_output_length,
     "level": level,
     "batch_size": batch_size,
     "pad_index": pad_index,
     "eos_index": eos_index,
     "batch_type": batch_type,
     #"device": device,
     "no_running_starts": no_running_starts,
     #"actor_id": i,
     }

    return parsed_config, actor_data_prefixes, actor_device_list

