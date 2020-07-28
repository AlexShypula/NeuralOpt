import pandas as pd
import sentencepiece as spm
import random
import os
import shutil
import re
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from os.path import dirname, join, basename
from stoke_preprocess import hash_file, mkdir, process_raw_assembly, merge_registers, stitch_together
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


@dataclass
class ParseOptions:
    path_to_destination_data: str = field(metadata=dict(args=["-dest", "--path_to_destination_data"]))
    path_to_source_data: str = field(metadata=dict(args=["-src", "--path_to_source_data"]))
    path_to_stats_csv: str = field(metadata=dict(args=["-stats_csv", "--path_to_stats_csv"]))
    path_to_model_data: str = field(metadata=dict(args=["-model_data_path", "--path_to_model_data"]))
    path_to_spm_model: str =  field(metadata=dict(args=["-spm_model", "--path_to_spm_model"]))
    n_threads: int = field(metadata=dict(args=["-n_threads"]), default=16)
    percent_train: float = field(metadata=dict(args=["-pct_train", "--percent_train"]), default=0.90)
    percent_val: float = field(metadata=dict(args=["-pct_val", "--percent_val"]), default=0.05)
    live_out_str: str = field(metadata=dict(args=["-live_out", "--live_out_str"]),
                                 default="{ %rax %rdx %rbx %rsp %rbp %r12 %r13 %r14 %r15 %xmm0 %xmm1 }")
    def_in_str: str = field(metadata=dict(args=["-def_in", "--def_in_str"]),
                    default="{ %rdx %rbx %rsp %rbp %rdi %r12 %r13 %r14 %r15 %xmm0 %xmm1 %mxcsr::rc[0] }")



def function_path_to_functions_folder(path: str):
    return dirname(path)


def function_path_to_binary_folder(path: str):
    return dirname(dirname(dirname(path)))


def function_path_to_function_file_name(path: str):
    return basename(path)


def function_file_name_to_testcase_file_name(filename: str):
    return re.sub("\.s", ".tc", filename)


def function_path_to_testcase_file_name(path: str):
    fun_file_name = function_path_to_function_file_name(path)
    testcase_file_name = function_file_name_to_testcase_file_name(fun_file_name)
    return testcase_file_name


def function_path_to_testcases(path: str, tc_folder: str = "testcases"):
    data_to_binary = function_path_to_binary_folder(path)
    testcase_file_name = function_path_to_testcase_file_name(path)
    return join(data_to_binary, tc_folder, testcase_file_name)


def remove_first_n_dirs(path: str, n_dirs_to_remove: int = 1):
    return join(*path.split("/")[n_dirs_to_remove:])


def function_path_to_unique_name(path: str):
    path = remove_first_n_dirs(path, n_dirs_to_remove=1)
    filename = function_path_to_function_file_name(path)
    filename = re.sub("\.s", "", filename)
    binary_folder_path = function_path_to_binary_folder(path)
    unique_id = join(binary_folder_path, filename)
    return re.sub("/", "_", unique_id)


def function_path_to_optimized_function(path: str, optimized_flag: str = "Og"):
    split_path = path.split("/")
    # -1 -> function name, -2 -> "functions", -3 -> "O0/Og/..." flag
    split_path[-3] = optimized_flag
    return join(*split_path)


def replace_first_n_dirs(path: str, path_to_destination_directory: str, n_dirs_to_remove: int = 1):
    destination_directory_to_target = remove_first_n_dirs(path)
    return join(path_to_destination_directory, destination_directory_to_target)


def individual_make_data(path_to_destination_data: str, path_to_source_data: str,
                         stats_dataframe: pd.DataFrame, index: int,
                         sent_piece_model: spm.SentencePieceProcessor,
                         live_out_str: str, def_in_str: str):

    data_path_to_function = stats_dataframe.iloc[index]["path_to_function"]
    unopt_cost = stats_dataframe.iloc[index]["unopt_unopt_cost"]
    opt_cost = stats_dataframe.iloc[index]["opt_unopt_cost"]

    data_path_to_optimized_function = function_path_to_optimized_function(data_path_to_function)
    data_path_to_function_folder = function_path_to_functions_folder(data_path_to_function)
    data_path_to_testcases = function_path_to_testcases(data_path_to_function)
    unique_name = function_path_to_unique_name(data_path_to_function)

    destination_path_to_function = replace_first_n_dirs(data_path_to_function, path_to_destination_data,
                                                        n_dirs_to_remove=1)
    destination_path_to_optimized_function = replace_first_n_dirs(data_path_to_optimized_function,
                                                                  path_to_destination_data, n_dirs_to_remove=1)
    destination_path_to_function_folder = replace_first_n_dirs(data_path_to_function_folder,
                                                               path_to_destination_data, n_dirs_to_remove=1)
    destination_path_to_testcases = replace_first_n_dirs(data_path_to_testcases, path_to_destination_data,
                                                         n_dirs_to_remove=1)

    if not os.path.exists(destination_path_to_function):
        shutil.copytree(join(path_to_source_data, data_path_to_function_folder), destination_path_to_function_folder)
    else:
        assert os.path.exists(destination_path_to_function)
    mkdir(dirname(destination_path_to_optimized_function))
    mkdir(destination_path_to_testcases)
    shutil.copy2(join(path_to_source_data, data_path_to_optimized_function), destination_path_to_optimized_function)
    shutil.copy2(join(path_to_source_data, data_path_to_testcases), destination_path_to_testcases)

    with open(join(path_to_source_data, data_path_to_function), "r") as f:
        raw_asbly = f.read()
        processed_asbly, _, _, _ = process_raw_assembly(raw_asbly)
        tokenized_asbly = merge_registers(sent_piece_model.EncodeAsPieces(processed_asbly.strip()))
        unopt_asbly_str = " ".join(tokenized_asbly)
        assembly_hash = hash_file(unopt_asbly_str)

    with open(data_path_to_optimized_function, "r") as f:
        raw_asbly = f.read()
        processed_asbly, _, _, _ = process_raw_assembly(raw_asbly)
        tokenized_asbly = merge_registers(sent_piece_model.EncodeAsPieces(processed_asbly.strip()))
        optimized_asbly_string = " ".join(tokenized_asbly)

    return unopt_asbly_str, optimized_asbly_string, \
    assembly_hash, {"base_asbly_path": destination_path_to_function,
                    "testcase_path": destination_path_to_testcases,
                    "O0_cost": unopt_cost,
                    "Og_cost": opt_cost,
                    "name": unique_name,
                    "cost_conf": {"def_in": def_in_str,
                                  "live_out": live_out_str,
                                  "distance": "hamming",
                                  "misalign_penalty": 1,
                                  "sig_penalty": "9999",
                                  "costfn": "100*correctness + latency + measured"}}


def individual_make_data_wrapper(arg_dict):
    return individual_make_data(**arg_dict)


def make_data(path_to_destination_data: str, path_to_source_data: str,
              stats_dataframe: pd.DataFrame, path_to_model_data: str, sent_piece_model: spm.SentencePieceProcessor,
              live_out_str: str, def_in_str: str, n_threads: int = 16, percent_train: float = 0.9,
              percent_val: float = 0.05, **kwargs):
    jobs = []
    for i in range(len(stats_dataframe)):
        arg_dict = {"path_to_destination_data": path_to_destination_data,
                    "path_to_source_data": path_to_source_data,
                    "stats_dataframe": stats_dataframe, "index": i, "sent_piece_model": sent_piece_model,
                    "live_out_str": live_out_str, "def_in_str": def_in_str}
        jobs.append(arg_dict)

    train_src = open(join(path_to_model_data, "train.src"), "w")
    train_tgt = open(join(path_to_model_data, "train.tgt"), "w")
    val_src = open(join(path_to_model_data, "val.src"), "w")
    val_tgt = open(join(path_to_model_data, "val.tgt"), "w")
    test_src = open(join(path_to_model_data, "test.src"), "w")
    test_tgt = open(join(path_to_model_data, "test.tgt"), "w")

    hash2metadata_dict = {}
    pbar = tqdm(total = len(stats_dataframe), smoothing = 0)

    for unopt_asbly, opt_asbly, asbly_hash, metadata_dict in ThreadPool(n_threads).imap(individual_make_data_wrapper,
                                                                                        jobs, chunksize=88):
        hash2metadata_dict[asbly_hash] = metadata_dict
        pbar.update()

        r = random.random()

        if r < percent_train:
            train_src.write(unopt_asbly + "\n")
            train_tgt.write(opt_asbly + "\n")
        elif r < (percent_train + percent_val):
            val_src.write(unopt_asbly + "\n")
            val_tgt.write(opt_asbly + "\n")
        else:
            test_src.write(unopt_asbly + "\n")
            test_tgt.write(opt_asbly + "\n")

    train_src.close()
    train_tgt.close()
    val_src.close()
    val_tgt.close()
    test_src.close()

    with open(join(path_to_model_data, "train_data.json"), "w") as fh:
        json.dump(hash2metadata_dict, fh, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    args.sent_piece_model = spm.SentencePieceProcessor()
    args.sent_piece_model.Load(args.path_to_spm_model)

    stats_dataframe = pd.read_csv(args.path_to_stats_csv)
    args.stats_dataframe = stats_dataframe[stats_dataframe["opt_unopt_correctness"] == "yes"][stats_dataframe["unopt_unopt_correctness"] == "yes"]
    args.stats_dataframe = args.stats_dataframe.reindex()

    make_data(**vars(args))

