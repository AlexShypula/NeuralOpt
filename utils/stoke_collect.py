
import json
from pymongo import MongoClient
from tqdm import tqdm
from typing import Dict
import sys
import os
import json
import subprocess



def collect_file_names(database: str, out_file_prefix: str, collection: str = "repos") -> None:
	"""
	collects the file names the specified database (i.e. the assembly file, the corresponding ELF file, as well as the corresponding hashes for both)

	database: database to query for results

	returns a dictionary in which the key is the full path to the assembly file and its value is a dictionary with the repo_path, original assembly_file name, corresponding ELF file name, and their respective hashes
	"""

	client= MongoClient()
	db_compile_results = client[database][collection]

	file_names_dictionary = {}

	for compile_result in tqdm(db_compile_results.find()):
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



def decompile_both(unopt_compile_path: str, opt_compile_path: str, unopt_data_dict: Dict[str, Dict], opt_data_dict: Dict[str, Dict] = None,  unopt_prefix = "Og", opt_prefix = "Og"):

	running_unopt_sha_set = set()

	for binary_identifier in tqdm(unopt_data_dict):
		if unopt_data_dict[binary_identifier]["assembly_sha"] not in running_unopt_sha_set:
			if opt_data_dict and binary_identifier in opt_data_dict:

				binary_path = binary_identifier[:-2] # based on the current way the collect script is written, last 2 chars will always be .s

				unopt_dict = unopt_data_dict[binary_identifier]
				opt_dict = opt_data_dict[binary_identifier]

				if not copy_and_decompile(unopt_dict, unopt_compile_path, binary_path, unopt_prefix):
					continue
				else:
					copy_and_decompile(opt_dict, opt_compile_path, binary_path, opt_prefix)




def copy_and_decompile(data_dict, compile_path, binary_path, optimization_prefix):
	try:
		pth = os.path.join([binary_path, optimization_prefix, "bin"])
		os.makedirs(pth)

		pth = os.path.join([binary_path, optimization_prefix, "functions"])
		os.makedirs(pth)

	except FileExistsError:
		print(f"path: {pth} already exists, moving to next binary")
		return False

	path_to_orig_bin = os.path.join(compile_path, data_dict["repo_path"], data_dict["ELF_sha"])
	path_to_local_bin = os.path.join([binary_path, optimization_prefix, "bin"])
	path_to_functions = os.path.join([binary_path, optimization_prefix, "functions"])
	subprocess.run(["cp", path_to_orig_bin, path_to_local_bin])
	subprocess.run(['stoke', 'extract', '-i', path_to_local_bin, "-o", path_to_functions])
	return True












