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
from typing import Set


@dataclass
class ParseOptions:
    path_to_destination_data: str = field(metadata=dict(args=["-dest", "--path_to_destination_data"]))
    path_to_source_data: str = field(metadata=dict(args=["-src", "--path_to_source_data"]))
    path_to_stats_csv: str = field(metadata=dict(args=["-stats_csv", "--path_to_stats_csv"]))
    path_to_model_data: str = field(metadata=dict(args=["-model_data_path", "--path_to_model_data"]))
    path_to_spm_model: str =  field(metadata=dict(args=["-spm_model", "--path_to_spm_model"]))
    path_to_train_list: str = field(metadata=dict(args=["-train_paths", "--path_to_train_paths"]))
    path_to_dev_list: str = field(metadata=dict(args=["-dev_paths", "--path_to_dev_paths"]))
    path_to_test_list: str = field(metadata=dict(args=["-test_paths", "--path_to_test_paths"]))
    n_threads: int = field(metadata=dict(args=["-n_threads"]), default=16)
    optimized_flag: str = field(metadata=dict(args=["-optim_flag", "--optimize_flag"]), default="Og")


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
    return "/".join(split_path)


def replace_first_n_dirs(path: str, path_to_destination_directory: str, n_dirs_to_remove: int = 1):
    destination_directory_to_target = remove_first_n_dirs(path, n_dirs_to_remove=n_dirs_to_remove)
    return join(path_to_destination_directory, destination_directory_to_target)


def individual_make_data(path_to_destination_data: str, path_to_source_data: str, dataframe_row: pd.core.series.Series,
                         sent_piece_model: spm.SentencePieceProcessor, optimized_flag: str,
                         ):

    data_path_to_function = dataframe_row["path_to_function"]
    unopt_cost = dataframe_row["unopt_unopt_cost"]
    opt_cost = dataframe_row["opt_unopt_cost"]
    def_in_str = dataframe_row["def_in"]
    live_out_str = dataframe_row["live_out"]
    heap_out = dataframe_row["heap_out"]

    data_path_to_optimized_function = function_path_to_optimized_function(data_path_to_function,
                                                                          optimized_flag=optimized_flag)
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
    mkdir(dirname(destination_path_to_testcases))
    shutil.copy2(join(path_to_source_data, data_path_to_optimized_function), destination_path_to_optimized_function)
    shutil.copy2(join(path_to_source_data, data_path_to_testcases), destination_path_to_testcases)

    with open(join(path_to_source_data, data_path_to_function), "r") as f:
        raw_asbly = f.read()
        processed_asbly, _, _, _ = process_raw_assembly(raw_asbly)
        tokenized_asbly = merge_registers(sent_piece_model.EncodeAsPieces(processed_asbly.strip()))
        unopt_asbly_str = " ".join(tokenized_asbly)
        assembly_hash = hash_file(unopt_asbly_str)

    with open(join(path_to_source_data, data_path_to_optimized_function), "r") as f:
        raw_asbly = f.read()
        processed_asbly, _, _, _ = process_raw_assembly(raw_asbly)
        tokenized_asbly = merge_registers(sent_piece_model.EncodeAsPieces(processed_asbly.strip()))
        optimized_asbly_string = " ".join(tokenized_asbly)

    path_to_binary_folder = function_path_to_binary_folder(data_path_to_function)

    return unopt_asbly_str, optimized_asbly_string, path_to_binary_folder, \
    assembly_hash, {"base_asbly_path": destination_path_to_function,
                    "testcase_path": destination_path_to_testcases,
                    "O0_cost": unopt_cost,
                    "Og_cost": opt_cost,
                    "name": unique_name,
                    "cost_conf": {"def_in": def_in_str,
                                  "live_out": live_out_str,
                                  "heap_out": heap_out,
                                  "distance": "hamming",
                                  "misalign_penalty": 1,
                                  "sig_penalty": "9999",
                                  "costfn": "100*correctness + latency + measured"}}


def individual_make_data_wrapper(arg_dict):
    return individual_make_data(**arg_dict)

'''(path_to_destination_data: str, path_to_source_data: str, dataframe_row: pd.core.series.Series,
                         sent_piece_model: spm.SentencePieceProcessor, optimized_flag: str,
                         ):'''

'''    args.train_paths = set([p.strip() for p in train_paths])
    args.dev_paths = set([p.strip() for p in dev_paths])
    args.test_paths = set([p.strip() for p in test_paths])'''

def make_data(path_to_destination_data: str, path_to_source_data: str,
              stats_dataframe: pd.DataFrame, path_to_model_data: str, sent_piece_model: spm.SentencePieceProcessor,
              train_paths: Set[str], dev_paths: Set[str], test_paths: Set[str], optimized_flag: str,
              n_threads: int = 16, **kwargs):
    jobs = []
    for _, row in stats_dataframe.iterrows():
        arg_dict = {"path_to_destination_data": path_to_destination_data,
                    "path_to_source_data": path_to_source_data,
                    "dataframe_row": row, "sent_piece_model": sent_piece_model,
                    "optimized_flag": optimized_flag}
        jobs.append(arg_dict)

    train_src = open(join(path_to_model_data, "train.src"), "w")
    train_tgt = open(join(path_to_model_data, "train.tgt"), "w")
    val_src = open(join(path_to_model_data, "val.src"), "w")
    val_tgt = open(join(path_to_model_data, "val.tgt"), "w")
    test_src = open(join(path_to_model_data, "test.src"), "w")
    test_tgt = open(join(path_to_model_data, "test.tgt"), "w")

    hash2metadata_dict = {}
    pbar = tqdm(total = len(stats_dataframe), smoothing = 0)

    for unopt_asbly, opt_asbly, path_to_binary_folder, asbly_hash, metadata_dict in ThreadPool(n_threads).imap(
                                                            individual_make_data_wrapper, jobs, chunksize=88):
        hash2metadata_dict[asbly_hash] = metadata_dict
        pbar.update()
        path_to_binary_folder = remove_first_n_dirs(path_to_binary_folder, 2)
        in_train = path_to_binary_folder in train_paths
        in_dev = path_to_binary_folder in dev_paths
        in_test = path_to_binary_folder in test_paths
        assert sum([in_train, in_dev, in_test])==1, "uh oh, the binary directory is either in none or more than one\n"\
                                "of the sets of train, dev, and test paths. \n" \
                                "the bin path is: {} and in_train is: {}, in_dev is: {}, and in_test is: {}".format(
                                path_to_binary_folder, in_train, in_dev, in_test)

        if in_train:
            train_src.write(unopt_asbly + "\n")
            train_tgt.write(opt_asbly + "\n")
        elif in_train:
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

    with open(join(path_to_model_data, "hash2metadata.json"), "w") as fh:
        json.dump(hash2metadata_dict, fh, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    args.sent_piece_model = spm.SentencePieceProcessor()
    args.sent_piece_model.Load(args.path_to_spm_model)

    # read in the files
    with open(args.path_to_train_list, "r") as train_fh, open(args.path_to_dev_list, "r") as dev_fh, \
            open(args.path_to_test_list, "r") as test_fh:
        train_paths = train_fh.readlines()
        dev_paths = dev_fh.readlines()
        test_paths = test_fh.readlines()

    # then strip \n and convert into hashtable
    args.train_paths = set([p.strip() for p in train_paths])
    args.dev_paths = set([p.strip() for p in dev_paths])
    args.test_paths = set([p.strip() for p in test_paths])

    stats_dataframe = pd.read_csv(args.path_to_stats_csv)
    args.stats_dataframe = stats_dataframe[stats_dataframe["opt_unopt_correctness"] == "yes"][stats_dataframe["unopt_unopt_correctness"] == "yes"]
    args.stats_dataframe = args.stats_dataframe.reindex()

    make_data(**vars(args))

