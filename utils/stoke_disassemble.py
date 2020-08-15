import json
from pymongo import MongoClient
from tqdm import tqdm
from typing import Dict
import sys
import os
import json
import subprocess
import shutil
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import re

COLLAPSE_PATTERN = re.compile("/\./")


def collapse_path(path_string):
    return COLLAPSE_PATTERN.sub("/", path_string)


@dataclass
class ParseOptions:
    binary_info_in_file: str = field(metadata=dict(args=["-bin_file_in", "--binary_info_in_file"]))
    path_to_bin_dir: str = field(metadata=dict(args=["-binary_dir", "--abs_path_to_binary_dir"]))
    path_to_disas_dir: str = field(metadata=dict(args=["-disas_dir", "--path_to_disas_dir"]))
    optimization_flag: str = field(metadata=dict(args=["-opt_flag", "--optimization_flag"]))
    successful_path_out_file: str = field(metadata=dict(args=["-successful_path_out", "--successful_path_out_file"]))
    n_workers: int = field(metadata=dict(args=["-n_workers", "--n_workers"]), default=1)
    debug: bool = field(metadata=dict(args=["-debug", "--debug_mode"]), default=False)


def load_and_disassemle(binary_info_in_file: str, path_to_bin_dir: str, path_to_disas_dir: str, optimization_flag: str,
                      successful_path_out_file: str, n_workers=1, debug=False) -> None:

    assert not os.path.exists(
        successful_path_out_file), "the out-file specified already exists, please change or delete"

    with open(binary_info_in_file, "r") as fh:
        binary_info_dict = json.load(fh)

    successful_paths = parallel_disassemble(binary_info_dict=binary_info_dict, path_to_bin_dir=path_to_bin_dir,
                                            path_to_disas_dir=path_to_disas_dir, optimization_flag=optimization_flag,
                                            n_workers=n_workers, debug=debug)

    with open(successful_path_out_file, "w") as fh:
        for p in successful_paths: 
            fh.write(p+"\n")

    return None


def parallel_disassemble(binary_info_dict: Dict[str, Dict], path_to_bin_dir: str,  path_to_disas_dir: str,
                         optimization_flag: str, n_workers=1, debug=False):
    running_hash_set = set()
    jobs_list = []
    for binary_identifier, binary_dictionary in tqdm(binary_info_dict.items()):
        file_hash = binary_dictionary["file_hash"]
        if file_hash not in running_hash_set:
            # returns basename or path without ending

            # same as the path to the binary; however, with the file-ending removed
            rel_path_to_bin_identifier = os.path.splitext(binary_identifier)[0]
            rel_path_to_bin_identifier = collapse_path(
                rel_path_to_bin_identifier)

            copy_and_disas_dict = {"path_to_disas_dir": path_to_disas_dir,
                                   "path_to_bin_dir": path_to_bin_dir,
                                   "rel_path_to_bin_identifier": rel_path_to_bin_identifier,
                                   "optimization_prefix": optimization_flag,
                                   "binary_dictionary": binary_dictionary}
            jobs_list.append(copy_and_disas_dict)

    jobs_list = jobs_list[:3000]
    successful_paths = []

    with tqdm(total=len(jobs_list), smoothing=0) as pbar:
        if debug:
            for bin_pth, rc, msg in map(copy_and_disas_wrapper, jobs_list):
                pbar.update()
                # if rc is False, print error
                if not rc:
                    print(
                        f"on {bin_pth} the process had error {msg} with code: {rc}")
                else:
                    successful_paths.append(bin_pth)
        else:
            for bin_pth, rc, msg in ThreadPool(n_workers).imap_unordered(copy_and_disas_wrapper, jobs_list):
                pbar.update()
                # if rc is False, print error
                if not rc:
                    print(
                        f"on {bin_pth} the process had error {msg} with code: {rc}")
                else:
                    successful_paths.append(bin_pth)

    return successful_paths


def copy_and_disas_wrapper(args):
    return copy_and_disas(**args)


def copy_and_disas(path_to_disas_dir: str, path_to_bin_dir: str, rel_path_to_bin_identifier: str,
                   optimization_prefix: str, binary_dictionary: str) -> (str, bool, str):
    try:
        lcl_bin_fldr = os.path.join(
            path_to_disas_dir, rel_path_to_bin_identifier, optimization_prefix, "bin")
        os.makedirs(lcl_bin_fldr)

        lcl_fun_fldr = os.path.join(
            path_to_disas_dir, rel_path_to_bin_identifier, optimization_prefix, "functions")
        os.makedirs(lcl_fun_fldr)

    except FileExistsError:
        err = f"path: {os.path.join(path_to_disas_dir, rel_path_to_bin_identifier, optimization_prefix)} already exists"
        return rel_path_to_bin_identifier, False, err

    path_to_orig_bin = os.path.join(
        path_to_bin_dir, binary_dictionary["repo_path"], binary_dictionary["file_hash"])
    if not os.path.exists(path_to_orig_bin):
        err = f"binary file: {path_to_orig_bin} does not exist"
        return rel_path_to_bin_identifier, False, err
    path_to_local_bin = os.path.join(
        lcl_bin_fldr, binary_dictionary["file_hash"])
    shutil.copy2(path_to_orig_bin, path_to_local_bin)
    try:
        p = subprocess.run(['stoke', 'extract', '-i', path_to_local_bin,
                            "-o", lcl_fun_fldr], capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, UnicodeDecodeError) as err:
        return rel_path_to_bin_identifier, False, err
    if p.returncode != 0:
        return rel_path_to_bin_identifier, False, p.stderr
    else:
        return rel_path_to_bin_identifier, True, "process exited normally"


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    load_and_disassemle(**vars(args))
