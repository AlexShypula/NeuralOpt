import csv
import sentencepiece as spm
import re
import regex
import datetime
from tqdm import tqdm
from os.path import join, isfile
from os import listdir
from multiprocessing.pool import ThreadPool
from stoke_preprocess import hash_file, mkdir, process_raw_assembly, merge_registers, stitch_together, FINDALL_FUNCTIONS_PATTERN
from typing import List, Set
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from stoke_test_costfn import StopWatch, strip_function_names

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

RUNTIME_SEARCH_REGEX = regex.compile("(?<=Runtime:\s+)[0-9\.]+")
THROUGHPUT_SEARCH_REGEX = regex.compile("(?<=Throughput:\s+)[0-9\.]+")

UNSUPPORTED_REGEX = re.compile("(call\w{0,1}|j\w{1,4}) (\%\w+)")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")

FIELDNAMES = ["unopt_assembly_string", "bpe_assembly_hash", "canonicalized_assembly_hash",
			  "path_to_binary", "path_binary_to_unopt_flag",
			  "path_opt_flag_to_function_dir_name", "function_file_name"]

@dataclass
class ParseOptions:
	path_list: str = field(metadata=dict(args=["-path_list", "--list_of_decompiled_binaries"]))
	unopt_prefix: str = "O0"
	opt_prefix: str = "Og"
	fun_dir_suff: str = field(metadata=dict(args=["-fun_dir_suff", "--functions_folder_name"]), default='functions')
	stats_csv: str = field(metadata=dict(args=["-stats_out", "--statistics_file_name"]), default='stats.csv')
	max_seq_len: int = field(metadata=dict(args=["-max_len", "--max_seq_len"]), default=None)
	spm_model_path: str = field(metadata=dict(args=["-spm_model_path", "--sent_piece_model_path"]), default=None)
	separator: str = ","
	n_workers: int = 8
	filter_unsupported: bool = field(metadata=dict(args=["-filter_unsupported", "--filter_unsupported"]), default=False)


def parallel_extract_asm(path_list: List[str],
					   spm_model_path: str,
					   unopt_prefix: str = "O0",
					   opt_prefix: str = "Og",
					   fun_dir_suff: str = "functions",
					   max_seq_len: int = 256,
					   stats_csv: str = "stats.csv",
					   separator: str = ",",
					   n_workers: int = 8,
					   filter_unsupported: bool = True,
					   ):

	stop_watch = StopWatch()
	stop_watch.start()

	stats_csv_fh = open(stats_csv, "w")
	field_names = FIELDNAMES
	print("field names for csv are {}".format(field_names))

	dict_writer = csv.DictWriter(stats_csv_fh,
								 fieldnames=field_names,
								 delimiter=separator,
								 quoting=csv.QUOTE_ALL)

	dict_writer.writeheader()

	asbly_hash_set = set()

	sent_piece = spm.SentencePieceProcessor()
	sent_piece.Load(spm_model_path)
	template_dict = {"path": None,
						"asbly_hash_set": asbly_hash_set,
					 	"spm_model": sent_piece,
						"fun_dir_suff":  fun_dir_suff,
						"unopt_prefix": unopt_prefix,
						"opt_prefix": opt_prefix,
					 	"max_seq_len": max_seq_len,
					 	"filter_unsupported": filter_unsupported,
					}

	jobs = []
	for path in path_list:
		job = template_dict.copy()
		job["path"] = path
		jobs.append(job)

	with tqdm(total=len(jobs), smoothing=0) as pbar:
		for csv_dict_list in ThreadPool(n_workers).imap_unordered(par_extract_asm_from_bin_directory, jobs):
			pbar.update()
			dict_writer.writerows(csv_dict_list)

	stats_csv_fh.close()
	stop_watch.stop()
	print(f"time it took to run the entire program was {datetime.timedelta(seconds = stop_watch.time)}")


def par_extract_asm_from_bin_directory(args_dict):
	return extract_asm_from_bin_directory(**args_dict)


def extract_asm_from_bin_directory(path: str,
						asbly_hash_set: List[str],
						spm_model: spm.SentencePieceProcessor,
						fun_dir_suff: str = "functions",
						unopt_prefix: str = "O0",
						opt_prefix: str = "Og",
						max_seq_len: int = None,
						filter_unsupported: bool = False):

	unopt_fun_dir = join(path, unopt_prefix, fun_dir_suff)
	opt_fun_dir = join(path, opt_prefix, fun_dir_suff)

	src_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(unopt_fun_dir, f))]
	tgt_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(opt_fun_dir, f))]

	csv_rows = []

	for fun_file in src_lst:

		# if there is no match, then skip it for processing
		if fun_file not in tgt_lst:
			continue

		# dict keys: unopt_assembly_string, bpe_assembly_hash, canonicalized_assembly_hash
		keep_flag, csv_row_dict = extract_tokenized_asm(fun_dir=unopt_fun_dir, fun_file=fun_file,
													   asbly_hash_set=asbly_hash_set, spm_model=spm_model,
													   filter_unsupported=filter_unsupported, max_seq_len=max_seq_len)

		# add both these guys to the hash set so we trim as much as possible
		asbly_hash_set.add(csv_row_dict["bpe_assembly_hash"])
		asbly_hash_set.add(csv_row_dict["canonicalized_assembly_hash"])

		if keep_flag:
			csv_row_dict["path_to_binary"] = path
			csv_row_dict["path_binary_to_unopt_flag"] = unopt_prefix
			csv_row_dict["path_opt_flag_to_function_dir_name"] = fun_dir_suff
			csv_row_dict["function_file_namme"] = fun_file
			csv_rows.append(csv_row_dict)

	return csv_rows


def extract_tokenized_asm(fun_dir: str, fun_file: str, asbly_hash_set: Set[str],
						spm_model: spm.SentencePieceProcessor, filter_unsupported: bool = False,
						max_seq_len: int = 256):

	path_to_function = join(fun_dir, fun_file)

	with open(path_to_function) as f:
		assembly = f.read()

	asbly, _, function_name_list, _ = process_raw_assembly(raw_assembly=assembly, preserve_fun_names=True, preserve_semantics=True)
	full_canon_asbly = strip_function_names(asbly, function_name_list)
	full_canon_assembly_hash = hash_file(full_canon_asbly) # used for deduplicating
	tokenized_asbly = merge_registers(spm_model.EncodeAsPieces(asbly.strip()))
	tokenized_asbly_string = " ".join(tokenized_asbly)
	bpe_hash = hash_file(tokenized_asbly_string) # has other purposes (i.e. use in training)

	# keys correspond to column names, values are the corresponding row values
	csv_row_dict = {"unopt_assembly_string": tokenized_asbly_string,
				   "bpe_assembly_hash": bpe_hash,
				   "canonicalized_assembly_hash": full_canon_assembly_hash}

	keep_flag = True
	if len(tokenized_asbly) > max_seq_len \
			or full_canon_assembly_hash in asbly_hash_set \
			or bpe_hash in asbly_hash_set:
		keep_flag = False
	elif filter_unsupported:
		match = UNSUPPORTED_REGEX.search(assembly)
		if match:
			keep_flag = False

	return keep_flag, csv_row_dict

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    with open(args.path_list) as f:
        path_list = f.readlines()
    path_list = [p.strip() for p in path_list]
    args.path_list = path_list
    parallel_extract_asm(**vars(args))
