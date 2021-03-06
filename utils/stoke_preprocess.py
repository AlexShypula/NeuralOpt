import hashlib
import re
import subprocess
import random
import os
import re
import sentencepiece as spm
import pickle
from os import listdir, makedirs
from os.path import isfile, join
from typing import List, Dict, Set
from collections import OrderedDict
from tqdm import tqdm
from collections import Counter
from multiprocessing.pool import Pool
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
#from stoke_test_costfn import strip_function_names


random.seed(15213)


@dataclass
class ParseOptions:
    path_to_bin: str = field(metadata=dict(args=["-path_to_bin", "--path_to_decompiled_binaries"]))
    path_list: str = field(metadata=dict(args=["-path_list", "--list_of_decompiled_binaries"]))
    unopt_prefix: str = "O0"
    opt_prefix: str = "Og"
    fun_dir: str = field(metadata=dict(args=["-fun_dir", "--functions_folder_name"]), default='functions')
    train_fldr: str = field(metadata=dict(args=["-train_fldr", "--train_working_folder_name"]), default='train')
    dev_fldr: str = field(metadata=dict(args=["-dev_fldr", "--dev_working_folder_name"]), default='dev')
    test_fldr: str = field(metadata=dict(args=["-test_fldr", "--test_working_folder_name"]), default='test')
    unmatched_fldr: str = field(metadata=dict(args=["-unmatched_fldr", "--unmatched_working_folder_name"]), default='unmatched')
    model_fldr: str = field(metadata=dict(args=["-model_fldr", "--bpe_folder_name"]), default='bpe')
    n_workers: int = 8
    n_splits: int = 16
    no_fun_names: bool = field(metadata=dict(args=["-no_fun_names", "--remove_fun_names"]), default = False)
    full_canonicalization: bool = field(metadata=dict(args=["-full_canon", "--fully_canonicalize_locations"]), default = False)
    percent_dev: float = field(metadata=dict(args=["-percent_dev", "--percent_dev"]), default = 0.05)
    percent_test: float = field(metadata=dict(args=["-percent_test", "--percent_test"]), default=0.05)
    path_to_initial_hashes: str = field(metadata=dict(args=["-path_to_in_hashes", "--path_to_init_hashes"]), default=None)
    path_to_output_hashes: str = field(metadata=dict(args=["-path_to_out_hashes", "--path_to_output_hashes"]), default=None)
    max_len: int = field(metadata=dict(args=["-max_len", "--maximum_assembly_length"]), default=512)


def strip_function_names(assembly_text: str, function_name_list: List[str]):
	for function_name in function_name_list:
		assembly_text = re.sub(f"\.{function_name}:[^\n]*\n", "", assembly_text)
		assembly_text = re.sub(f"\.size\s+{function_name},\s+.-.*", "", assembly_text)
	return assembly_text


def hash_file(file_string: str, encoding: str = "utf-8") -> str:
    m = hashlib.sha512()
    m.update(bytes(file_string, encoding))
    return m.hexdigest()


def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)


def parallel_pipeline(path_to_bin: str,
                path_list: List[str],
                unopt_prefix: str = "O0",
                opt_prefix: str = "Og",
                fun_dir: str = "functions",
                train_fldr: str = "train",
                dev_fldr:str = "dev",
                test_fldr: str = "test",
                unmatched_fldr: str = "unmatched",
                model_fldr: str = "bpe",
                n_workers: int = 8,
                n_splits: int = 16,
                no_fun_names: bool = False,
                full_canonicalization: bool = False,
                percent_dev : float= 0.05,
                percent_test: float = 0.05,
                initial_hashes: Set = None,
                max_len: int = 512, **kwargs) -> Set[str]:
    # returns the set of hashes


    assert n_splits % n_workers == 0
    mkdir(train_fldr)
    mkdir(dev_fldr)
    mkdir(test_fldr)
    mkdir(unmatched_fldr)

    args_dict = {"path_to_bin": path_to_bin,
                "path_list": None,
                "unopt_prefix": unopt_prefix,
                "opt_prefix": opt_prefix,
                "fun_dir": fun_dir,
                "train_fldr": train_fldr,
                "dev_fldr": dev_fldr,
                "test_fldr": test_fldr,
                "unmatched_fldr": unmatched_fldr,
                "suffix": None,
                "preserve_fun_names": not no_fun_names,
                "preserve_semantics": not full_canonicalization,
                "percent_dev": percent_dev,
                "percent_test": percent_test
                }

    jobs = []
    job_chunk_size = (len(path_list) // n_splits) + 1
    for enum, index in enumerate(range(0, len(path_list), job_chunk_size)):
        path_list_chunk = path_list[index:index+job_chunk_size]
        job_dict = args_dict.copy()
        job_dict["suffix"] = enum
        job_dict["path_list"] = path_list_chunk
        jobs.append(job_dict)

    tot_missing = 0
    tot_dups = 0
    tot_success = 0
    for n_miss, n_dup, n_succ in Pool(n_workers).imap_unordered(parallel_process_wrapper, jobs):
        tot_missing += n_miss
        tot_dups += n_dup
        tot_success += n_succ

    n_functions = tot_missing + tot_dups + tot_success
    print("-"*60 + "\n\n" + "-"*60 + "\n\n" + "-"*60 )
    print(f"total: percent of function files with missing pair is {(tot_missing / n_functions) * 100:.2f}%")
    print(f"total: percent of function files duplicated is {(tot_dups / n_functions) * 100:.2f}%")
    print(f"total: percent of function files available {(tot_success / n_functions) * 100:.2f}%")
    print(f"total: number of successful source target pairs of assembly functions is {tot_success}")

    print("Now training BPE Model ..... ")

    spm_train(train_folder=train_fldr, model_folder=model_fldr)

    print("Building list of jobs for processing the training set ... ")

    args_dict = {"in_src_file": None,
                 "in_tgt_file": None,
                 "out_src_file": None,
                 "out_tgt_file": None,
                 "spm_model_pth": join(model_fldr, "bpe_1000.model"),
                 "threshold": 2056,
                 "hashes": set()}
    jobs = []
    for i in range(n_splits):
        job_dict = args_dict.copy()
        job_dict["in_src_file"] = join(train_fldr, f"train_{i}.src")
        job_dict["in_tgt_file"] = join(train_fldr, f"train_{i}.tgt")
        job_dict["out_src_file"] = join(train_fldr, f"train_{i}_fnl.src")
        job_dict["out_tgt_file"] = join(train_fldr, f"train_{i}_fnl.tgt")
        jobs.append(job_dict)

    print("Now processing all training files with bpe ...")

    if not initial_hashes:
        running_hashes = set()
    else:
        running_hashes = initial_hashes
    for tr_hashes, _ in Pool(n_workers).imap_unordered(parallel_bpe_process_wrapper, jobs):
        running_hashes.update(tr_hashes)

    unmatched_dups = 0
    # process unmatched
    for i in range(n_splits):
        hashes, dups = bpe_process_single(in_file= join(unmatched_fldr, f"unmatched_{i}.src"),
                                          out_file=join(unmatched_fldr, f"unmatched_{i}_fnl.src"),
                                          spm_model_pth=join(model_fldr, "bpe_1000.model"), threshold = max_len,
                                          hashes = running_hashes)
        running_hashes.update(hashes)
        unmatched_dups += dups

    dev_dups = 0
    # process dev set
    for i in range(n_splits):
        hashes, dups = bpe_process(join(dev_fldr, f"dev_{i}.src"), join(dev_fldr, f"dev_{i}.tgt"),
                                   join(dev_fldr, f"dev_{i}_fnl.src"), join(dev_fldr, f"dev_{i}_fnl.tgt"),
                                   spm_model_pth = join(model_fldr, "bpe_1000.model"), threshold = max_len,
                                   hashes = running_hashes)
        running_hashes.update(hashes)
        dev_dups += dups

    # process test_set
    test_dups = 0
    for i in range(n_splits):
        hashes, dups = bpe_process(join(test_fldr, f"test_{i}.src"), join(test_fldr, f"test_{i}.tgt"),
                                   join(test_fldr, f"test_{i}_fnl.src"), join(test_fldr, f"test_{i}_fnl.tgt"),
                                   spm_model_pth = join(model_fldr, "bpe_1000.model"), threshold = max_len,
                                   hashes = running_hashes)
        running_hashes.update(hashes)
        test_dups += dups


    print(f"total number of unique functions (source and target) in train, val, test combined is {len(running_hashes)}")
    print(f"total number of dev duplicates is {dev_dups}")
    print(f"total number of test duplicates is {test_dups}")
    print(f"total number of unmatched function duplicates is {unmatched_dups}")

    print("Now merging all the files and recording vocab...")

    tok_cts = merge_all_files(fldr_pth=train_fldr, n_splits=n_splits, out_src="train.src", out_tgt="train.tgt",
                    mode="train", count=True)

    merge_all_files(fldr_pth=dev_fldr, n_splits=n_splits, out_src="dev.src", out_tgt="dev.tgt",
                    mode="dev", count=False)
    merge_all_files(fldr_pth=test_fldr, n_splits=n_splits, out_src="test.src", out_tgt="test.tgt",
                    mode="test", count=False)
    merge_all_files(fldr_pth=unmatched_fldr, n_splits=n_splits, out_src="unmatched.src", out_tgt=None,
                    mode="unmatched", count=False)


    print("Writing out the vocab ...")
    with open(join(model_fldr, "vocab.txt"), "w") as f:
        for tok, _ in tok_cts.most_common():
            if tok != " ":
                f.write(tok + "\n")

    print("now deleting all intermediate files")
    for split in ["train", "dev", "test", "unmatched"]:
        for file_ending in ["src", 'tgt']:
            for i in range(n_splits):
                file = f"{split}_{i}.{file_ending}"
                if os.path.exists(file):
                    os.remove(file)
                    print(f"deleted {file}")
                file = f"{split}_{i}_fnl.{file_ending}"
                if os.path.exists(file):
                    os.remove(file)
                    print(f"deleted {file}")

    print("Done ! Nice !!")
    return running_hashes


def merge_all_files(fldr_pth: str, n_splits: int, out_src: str, out_tgt: str = None,
                    mode: str = "train", count = False):
    global_counter = Counter()
    src = open(join(fldr_pth, out_src), "w")
    if out_tgt:
        tgt = open(join(fldr_pth, out_tgt), "w")
    for i in range(n_splits):
        fn = f"{mode}_{i}_fnl.src"
        with open(join(fldr_pth, fn)) as f:
            tmp = f.read()
            src.write(tmp)
            if count:
                c = Counter(tmp.split())
                global_counter = global_counter + c
        if out_tgt:
            fn = f"{mode}_{i}_fnl.tgt"
            with open(join(fldr_pth, fn)) as f:
                tmp = f.read()
                tgt.write(tmp)
                if count:
                    c = Counter(tmp.split())
                    global_counter = global_counter + c

    return global_counter

# bpe_process(in_src_file: str, in_tgt_file: str, out_src_file: str, out_tgt_file: str, spm_model_pth: str, threshold = 200, hashes = None):


# process_all("stoke", path_list, train_fldr = "stoke/train/", dev_fldr = "stoke/dev/", test_fldr = "stoke/test/")

def parallel_process_wrapper(args: Dict[str, str]):
    return process_all(**args)

def process_all(path_to_bin: str,
                path_list: List[str],
                unopt_prefix: str = "O0",
                opt_prefix: str = "Og",
                fun_dir: str = "functions",
                train_fldr: str = "train",
                dev_fldr:str = "dev",
                test_fldr: str = "test",
                unmatched_fldr: str = "unmatched_fldr",
                suffix: str = "0",
                preserve_fun_names: bool = True,
                preserve_semantics: bool = True,
                percent_dev: float = 0.05,
                percent_test: float = 0.05):

    src_shas = set()
    tgt_shas = set()
    n_missing_pair = 0
    n_dups = 0
    n_empty = 0
    n_success = 0
    # breakpoint()
    # mkdir(train_fldr)
    # mkdir(dev_fldr)
    # mkdir(test_fldr)

    tr_src_f = open(join(train_fldr, f"train_{suffix}.src"), "w")
    tr_tgt_f = open(join(train_fldr, f"train_{suffix}.tgt"), "w")
    dev_src_f = open(join(dev_fldr, f"dev_{suffix}.src"), "w")
    dev_tgt_f = open(join(dev_fldr, f"dev_{suffix}.tgt"), "w")
    test_src_f = open(join(test_fldr, f"test_{suffix}.src"), "w")
    test_tgt_f = open(join(test_fldr, f"test_{suffix}.tgt"), "w")

    unmatched_src_f = open(join(unmatched_fldr, f"unmatched_{suffix}.src"), "w")

    for path in tqdm(path_list, position=int(suffix), desc = f"job {suffix}", leave=True):
        unopt_fun_dir = join(path_to_bin, path, unopt_prefix, fun_dir)
        opt_fun_dir = join(path_to_bin, path, opt_prefix, fun_dir)
        src_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(unopt_fun_dir, f))]
        tgt_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(opt_fun_dir, f))]
        if src_lst == []:
            n_empty += 1
        for f in src_lst:

            src_pth = join(unopt_fun_dir, f)
            with open(src_pth) as file_obj:
                src_text = file_obj.read()
                src_hash = hash_file(src_text)
            if src_hash in src_shas:
                n_dups += 1
                continue
            src_shas.add(src_hash)
            src_asbly, _, _, _ = process_raw_assembly(raw_assembly=src_text,
                                                      preserve_fun_names=preserve_fun_names,
                                                      preserve_semantics=preserve_semantics)
            if f not in tgt_lst:
                unmatched_src_f.write(src_asbly + "\n")
                n_missing_pair += 1
                continue

            tgt_pth = join(opt_fun_dir, f)
            with open(tgt_pth) as file_obj:
                tgt_text = file_obj.read()
                tgt_hash = hash_file(tgt_text)
            if tgt_hash in tgt_shas:
                n_dups += 1
                continue
            tgt_shas.add(tgt_hash)

            tgt_asbly, _, _, _ = process_raw_assembly(raw_assembly=tgt_text,
                                                      preserve_fun_names=preserve_fun_names,
                                                      preserve_semantics=preserve_semantics)
            rn = random.random()
            if rn < (1 - percent_dev - percent_test):
                tr_src_f.write(src_asbly+"\n")
                tr_tgt_f.write(tgt_asbly+"\n")
            elif rn < (1 - percent_test):
                dev_src_f.write(src_asbly + "\n")
                dev_tgt_f.write(tgt_asbly + "\n")
            else:
                test_src_f.write(src_asbly + "\n")
                test_tgt_f.write(tgt_asbly + "\n")
            n_success += 1
    n_functions = n_missing_pair + n_dups +  n_success
    print(f"job: {suffix}, percent of binaries with no functions is {(n_empty / len(path_list))*100:.2f}%")
    print(f"job: {suffix}, percent of function files with missing pair is {(n_missing_pair / n_functions)*100:.2f}%")
    print(f"job: {suffix}, percent of function files duplicated is {(n_dups / n_functions) * 100:.2f}%")
    print(f"job: {suffix}, percent of function files available {(n_success / n_functions) * 100:.2f}%")
    print(f"job: {suffix}, total number of successful source target pairs of assembly functions is {n_success}")
    tr_src_f.close()
    tr_tgt_f.close()
    dev_src_f.close()
    dev_tgt_f.close()
    test_src_f.close()
    test_tgt_f.close()

    return n_missing_pair, n_dups, n_success

def spm_train(train_folder:str, model_folder:str):
    mkdir(model_folder)
    in_f = join(train_folder, "train*")
    out_f = join(train_folder, "bpe_file.txt")
    # with open(out_f, "w") as file:
    #     subprocess.run(["cat", in_f,], shell=True, stdout=file)
    cat_str = f"cat {in_f} > {out_f}"
    subprocess.run(cat_str, shell=True)
    train_string = f"--input={out_f} --model_prefix={join(model_folder, 'bpe_1000')} --vocab_size=1000 --user_defined_symbols='</n>,</->,</n>,' --character_coverage=1.0 --model_type=bpe --max_sentence_length=10000"
    spm.SentencePieceTrainer.train(train_string)
    # #spm.SentencePieceTrainer.train(input=out_f, model_prefix='bpe_500', vocab_size=500,
    #                                user_defined_symbols=['</n>', '</->'], character_coverage=1.0,
    #                                model_type="bpe", max_sentence_length=10000)


def stitch_together(string):
    l = string.split()
    return ''.join(l).replace('▁', ' ').replace('</n>', '\n').replace('</->', '_')

def merge_registers(a):
    '''
    merge back split registers
    '''
    res = []
    last = a[0]
    for current in a[1:]:
        if last == None:
            last = current
            continue
        if last != '▁%' and last != '%':
            res.append(last)
            last = current
        else:
            res.append(''.join([last, current]))
            last = None
    if last != '▁%' and last != '%' and last != None:
        res.append(last)
    return res

def parallel_bpe_process_wrapper(args_dict: Dict[str, str]):
    return bpe_process(**args_dict)

def get_asbly_function_name(assembly: str):
    return re.search("(?<=(\.))[^:]+", assembly).group()

def get_canonicalized_hash(assembly: str):
    function_name = get_asbly_function_name(assembly=assembly)
    cleaned_asm = strip_function_names(assembly)
    h = hash_file(cleaned_asm.strip())
    return h



def bpe_process(in_src_file: str, in_tgt_file: str, out_src_file: str, out_tgt_file: str, spm_model_pth: str, threshold = 200, hashes = None):
    job_no = re.findall("\d|$", in_src_file)[0]
    dups = 0
    sent_piece = spm.SentencePieceProcessor()
    sent_piece.Load(spm_model_pth)

    with open(in_src_file) as f:
        src_data = f.readlines()
    with open(in_tgt_file) as f:
        tgt_data = f.readlines()
    src_out = open(out_src_file, "w+")
    tgt_out = open(out_tgt_file, "w+")
    for i, src_asbly in enumerate(tqdm(src_data, position=int(job_no), desc = f"job no: {job_no}", leave=True)):
        if type(hashes) == type(set()):
            src_hash = get_canonicalized_hash(src_asbly.strip())
            if src_hash in hashes:
                dups += 1
                continue
            hashes.add(src_hash)
        src_tok = merge_registers(sent_piece.EncodeAsPieces(src_asbly.strip()))
        if len(src_tok) < threshold:
            if type(hashes) == type(set()):
                tgt_hash = get_canonicalized_hash((tgt_data[i].strip()))
                if tgt_hash in hashes:
                    dups+=1
                    continue
                hashes.add(tgt_hash)
            src_str = " ".join(src_tok)#.replace('▁', ' ')
            src_out.write(src_str+"\n")
            tgt_tok = merge_registers(sent_piece.EncodeAsPieces(tgt_data[i].strip()))
            tgt_str = " ".join(tgt_tok)#.replace('▁', ' ')
            tgt_out.write(tgt_str+"\n")
    src_out.close()
    tgt_out.close()
    return (hashes, dups) if hashes else (set(), dups)

def bpe_process_single(in_file: str, out_file: str, spm_model_pth: str, threshold = 200, hashes = None):
    job_no = re.findall("\d|$", in_file)[0]
    dups = 0
    sent_piece = spm.SentencePieceProcessor()
    sent_piece.Load(spm_model_pth)

    with open(in_file) as f:
        data = f.readlines()
    out_file_handle = open(out_file, "w+")

    for i, asbly in enumerate(tqdm(data, position=int(job_no), desc = f"job no: {job_no}", leave=True)):
        if type(hashes) == type(set()):
            asbly_hash = get_canonicalized_hash(asbly.strip())
            if asbly_hash in hashes:
                dups += 1
                continue
            hashes.add(asbly_hash)
        tokenized_asbly = merge_registers(sent_piece.EncodeAsPieces(asbly.strip()))
        if len(tokenized_asbly) < threshold:
            tokenized_string = " ".join(tokenized_asbly)
            out_file_handle.write(tokenized_string+"\n")

    out_file_handle.close()
    return (hashes, dups) if hashes else (set(), dups)


METADATA_SPLIT_PATTERN = re.compile("(?=# Text)")
FINDALL_FUNCTIONS_PATTERN = re.compile("(?<=.type ).*?(?=, @function)")
COMMENT_PATTERN = re.compile("#.*?(?=\n)")
WHITESPACE_PATTERN = re.compile("\n+")
FINDALL_LOCATIONS_PATTERN = re.compile("\..*?(?=:|\s)")

NEW_LINE_PATTERN_REPLACE = re.compile("\n")
UNDERSCORE_PATTERN_REPLACE = re.compile("_")
NEW_LINE_PATTERN_UNDO = re.compile("</n>")
UNDERSCORE_PATTERN_UNDO = re.compile("</->")

def _spec_char_rep(assembly:str):
    assembly = NEW_LINE_PATTERN_REPLACE.sub("</n>", assembly)
    assembly = UNDERSCORE_PATTERN_REPLACE.sub("</->", assembly)
    return assembly

def _spec_char_undo(assembly:str):
    assembly = NEW_LINE_PATTERN_UNDO.sub("\n", assembly)
    assembly = UNDERSCORE_PATTERN_UNDO.sub("_", assembly)
    return assembly


def process_raw_assembly(raw_assembly: str, preserve_fun_names: bool = True, preserve_semantics: bool = True):
    metadata, assembly = _split_metadata(raw_assembly)
    if preserve_fun_names:
        function_list = FINDALL_FUNCTIONS_PATTERN.findall(metadata)
    else:
        function_list = []
    assembly, orig2canon_loc_dict = _process_assembly(assembly, function_list, preserve_semantics)
    assembly = _spec_char_rep(assembly.strip()) # remove leading and training \n
    return assembly, metadata, function_list, orig2canon_loc_dict

def _process_assembly(assembly: str, function_list: List[str], preserve_semantics: bool):
    no_comments = COMMENT_PATTERN.sub("", assembly)
    no_extra_space = WHITESPACE_PATTERN.sub("\n", no_comments)
    clean_assembly, orig2canon_loc_dict = _canonicalize_labels(no_extra_space, function_list, preserve_semantics)
    return clean_assembly, orig2canon_loc_dict

def _split_metadata(raw_assembly:str):
    metadata, assembly = METADATA_SPLIT_PATTERN.split(raw_assembly, maxsplit=1)
    return metadata, assembly

def _canonicalize_labels(assembly: str, function_list: List[str], preserve_semantics: bool = True):
    raw_locs = FINDALL_LOCATIONS_PATTERN.findall(assembly)
    # make a list of the locations that we'll keep
    kept_locs = [".size"]
    for fun in function_list:
        kept_locs.append("."+fun)
        kept_locs.append(".-" + fun)
    # get all idiosyncratic locations to replace
    idiosyn_locs = [l for l in OrderedDict.fromkeys(raw_locs)
                               if l not in kept_locs]
    # canonicalized locations starting from 1
    if preserve_semantics:
        canon_locs = [".L"+ str(i+1) for i in range(len(idiosyn_locs))]
    else:
        canon_locs = [".LOC"] * len(idiosyn_locs)
    idiosyn2canon = {idiosyn: canon for idiosyn, canon in zip(idiosyn_locs, canon_locs)}
    for idiosyn, canon in idiosyn2canon.items():
        # replace all occurrences
        assembly = re.sub(idiosyn, canon, assembly)
    return assembly, idiosyn2canon


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    if args.path_to_initial_hashes:
        with open(args.path_to_initial_hashes, "rb") as fh:
            args.initial_hashes = pickle.load(fh)
    else:
        args.initial_hashes = None
    with open(args.path_list) as f:
        path_list = f.readlines()
    path_list = [p.strip() for p in path_list]
    args.path_list = path_list
    out_hashes = parallel_pipeline(**vars(args))
    with open(args.path_to_output_hashes, "wb") as fh:
        pickle.dump(out_hashes, file = fh)

