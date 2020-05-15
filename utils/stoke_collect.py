import json
from pymongo import MongoClient
from tqdm import tqdm
from typing import Dict
import sys
import os
import json
import subprocess
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import re

COLLAPSE_PATTERN = re.compile("/\./")
def collapse_path(path_string): 
    return COLLAPSE_PATTERN.sub("/", path_string)
    


@dataclass
class ParseOptions:

	unopt_bin: str = field(metadata=dict(args=["-unopt_bin", "--unoptimized_binary_dir"]))
	opt_bin: str = field(metadata=dict(args=["-opt_bin", "--optimized_binary_dir"]))
	unopt_meta: str = field(metadata=dict(args=["-unopt_meta", "--unoptimized_bin_metadata"]))
	opt_meta: str = field(metadata=dict(args=["-opt_meta", "--optimized_bin_metadata"]))
	n_workers: int = 1
	unopt_flag: str = "O0"
	opt_flag: str = "Og"
	o: str = field(metadata=dict(args=["-o", "--out_file_name"]), default='succcessful_paths.txt')
	out_dir: str = field(metadata=dict(args=["-out_dir", "--out_binary_directory"]), default='processed_binaries')


def collect_file_names(database: str, out_file_prefix: str, collection: str = "repos") -> None:
	"""
	collects the file names the specified database (i.e. the assembly file, the corresponding ELF file, as well as the corresponding hashes for both)

	database: database to query for results

	returns a dictionary in which the key is the full path to the assembly file and its value is a dictionary with the repo_path, original assembly_file name, corresponding ELF file name, and their respective hashes
	"""

	client= MongoClient()
	db_compile_results = client[database][collection]

	file_names_dictionary = {}
	results = db_compile_results.find()
	total = results.count()

	for compile_result in tqdm(results, total = total):
		if compile_result["num_binaries"] > 0:
			for makefile in compile_result["makefiles"]:
				# ensure the makefile didn't fail or return no binaries
				if makefile["success"] == True and makefile["binaries"] != [] and makefile["sha256"] != []:
					directory = makefile["directory"]
					orig_files = makefile["binaries"]
					sha256_files = makefile['sha256']
					repo_path = "/".join([compile_result["repo_owner"],compile_result["repo_name"]])

					file2sha = {file: sha for file, sha in zip(orig_files, sha256_files)}

					for file_name in orig_files:
						if file_name[-2:] == ".s":
							shared_name = file_name[:-2]
							if (shared_name + ".o") in orig_files:
								ELF = shared_name + ".o"
							elif shared_name in orig_files:
								ELF = shared_name
							else:
								print(f"assembly file: {file_name}'s corresponding ELF could not be found in the following list {orig_files}", file=sys.stderr)
								continue
							identifier = "/".join([repo_path, directory, file_name])
							file_names_dictionary[identifier] = {	"repo_path": repo_path,
																	"directory": directory,
																	"assembly_file": file_name,
																	"ELF_file": ELF,
																	"assembly_sha": file2sha[file_name],
																	"ELF_sha": file2sha[ELF]
																	}


	with open(out_file_prefix + '.json', 'w') as f:
		json.dump(file_names_dictionary, f, indent = 4)


def decompile_both(unopt_compile_path: str, opt_compile_path: str, unopt_data_dict: Dict[str, Dict], opt_data_dict: Dict[str, Dict] = None,  unopt_prefix = "O0", opt_prefix = "Og", res_folder = "processed_binaries"):

	running_unopt_sha_set = set()
	successful_paths = []

	for binary_identifier in tqdm(unopt_data_dict):
		asbly_hash = unopt_data_dict[binary_identifier]["assembly_sha"]
		if asbly_hash not in running_unopt_sha_set:
			running_unopt_sha_set.add(asbly_hash)
			if opt_data_dict and binary_identifier in opt_data_dict:

				unopt_dict = unopt_data_dict[binary_identifier]
				opt_dict = opt_data_dict[binary_identifier]

				# based on the current way the collect script is written, last 2 chars will always be .s
				binary_path = collapse_path(binary_identifier[:-2])

				rc, err_msg =  copy_and_decompile(unopt_dict, unopt_compile_path, res_folder, binary_path, unopt_prefix)
				if not rc:
					print(f"on {binary_path} the process had error {err_msg} with code: {True}")
					continue
				rc, err_msg = copy_and_decompile(opt_dict, opt_compile_path, res_folder, binary_path, opt_prefix)
				if not rc:
					print(f"on {binary_path} the process had error {err_msg} with code: {True}")
				else:
					successful_paths.append(binary_path)

	return successful_paths


def parallel_decompile(unopt_compile_path: str, opt_compile_path: str, unopt_data_dict: Dict[str, Dict], opt_data_dict: Dict[str, Dict] = None,  unopt_prefix = "O0", opt_prefix = "Og", n_workers = 16, res_folder = "processed_binaries"):
	running_unopt_sha_set = set()

	jobs_list = []
	for binary_identifier in tqdm(unopt_data_dict):
		asbly_sha = unopt_data_dict[binary_identifier]["assembly_sha"]
		if asbly_sha not in running_unopt_sha_set:
			if opt_data_dict and binary_identifier in opt_data_dict:

				# based on the current way the collect script is written, last 2 chars will always be .s
				binary_path = collapse_path(binary_identifier[:-2])

				unopt_dict = unopt_data_dict[binary_identifier]
				opt_dict = opt_data_dict[binary_identifier]
				copy_and_decompile_dict = {"binary_path": binary_path,
										   "result_folder": res_folder,
										   "unopt_prefix": unopt_prefix,
										   "opt_prefix": opt_prefix,
										   "unopt_compile_path": unopt_compile_path,
										   "opt_compile_path": opt_compile_path,
										   "unopt_dict": unopt_dict,
										   "opt_dict": opt_dict}
				jobs_list.append(copy_and_decompile_dict)

	successful_paths = []
	with tqdm(total=len(jobs_list), smoothing=0) as pbar:
		for bin_pth, msg, rc in ThreadPool(n_workers).imap_unordered(run_dual_cpy_decompile, jobs_list):
			pbar.update()
			# if rc is False, print error
			if not rc:
				print(f"on {bin_pth} the process had error {msg} with code: {rc}")
			else:
				successful_paths.append(bin_pth)

	return successful_paths


def run_dual_cpy_decompile(copy_and_decompile_dict):

	rc, err_msg = copy_and_decompile(copy_and_decompile_dict["unopt_dict"],
										copy_and_decompile_dict["unopt_compile_path"],
										copy_and_decompile_dict["result_folder"],
										copy_and_decompile_dict["binary_path"],
										copy_and_decompile_dict["unopt_prefix"]
									)
	# if the rc is False then return with err
	if not rc:
		return copy_and_decompile_dict["binary_path"], err_msg, False

	rc, err_msg =  copy_and_decompile(copy_and_decompile_dict["opt_dict"],
										copy_and_decompile_dict["opt_compile_path"],
										copy_and_decompile_dict["result_folder"],
										copy_and_decompile_dict["binary_path"],
										copy_and_decompile_dict["opt_prefix"]
									)
	# if the rc is False then return with err
	if not rc:
		return copy_and_decompile_dict["binary_path"], err_msg, False

	else:
		return copy_and_decompile_dict["binary_path"], "success", True


def copy_and_decompile(data_dict, compile_path, result_folder, binary_path, optimization_prefix):
	try:
		lcl_bin_fldr = os.path.join(result_folder, binary_path, optimization_prefix, "bin")
		os.makedirs(lcl_bin_fldr)

		lcl_fun_fldr = os.path.join(result_folder, binary_path, optimization_prefix, "functions")
		os.makedirs(lcl_fun_fldr)

	except FileExistsError:
		err = f"path: {os.path.join(result_folder, binary_path, optimization_prefix)} already exists"
		return False, err

	path_to_orig_bin = os.path.join(compile_path, data_dict["repo_path"], data_dict["ELF_sha"])
	path_to_local_bin = os.path.join(lcl_bin_fldr, data_dict["ELF_sha"])
	# path_to_functions = os.path.join([binary_path, optimization_prefix, "functions"])
	try:
		p = subprocess.run(["cp", path_to_orig_bin, path_to_local_bin], capture_output=True, text=True, timeout=500)
	except subprocess.TimeoutExpired as err:
		return False, err
	if p.returncode == 0:
		try:
			p = subprocess.run(['stoke', 'extract', '-i', path_to_local_bin, "-o", lcl_fun_fldr], capture_output=True, text=True, timeout=500)
		except subprocess.TimeoutExpired as err:
                    return False, err
	if p.returncode!=0:
		return False, p.stderr
	else:
		return True, "process exited normally"


if __name__ == "__main__":
	parser = ArgumentParser(ParseOptions)
	print(parser.parse_args())
	args = parser.parse_args()

	with open(args.unopt_meta, 'r') as f:
		unopt_metadata = json.load(f)

	with open(args.opt_meta, 'r') as f:
		opt_metadata = json.load(f)

	if args.n_workers < 2:
		successful_paths = decompile_both(
			args.unopt_bin,
			args.opt_bin,
			unopt_metadata,
			opt_metadata,
			unopt_prefix = args.unopt_flag,
			opt_prefix = args.opt_flag,
			res_folder=args.out_dir
		)
	else:
		successful_paths = parallel_decompile(
			args.unopt_bin,
			args.opt_bin,
			unopt_metadata,
			opt_metadata,
			unopt_prefix=args.unopt_flag,
			opt_prefix=args.opt_flag,
			n_workers=args.n_workers,
			res_folder=args.out_dir
		)
	with open(args.o, "w+") as f:
		for path in successful_paths:
			full_pth = os.path.join(args.out_dir, path)
			f.write(full_pth + "\n")












