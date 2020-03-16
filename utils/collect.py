from pymongo import MongoClient
from tqdm import tqdm
from typing import Dict, List
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#import chunk
from .chunk import chunk_assembly, function_names, write_to_csv



def collect_file_names(database: str, collection: str = "repos") -> Dict[str, Dict]: 
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
								print(f"assembly file: {file_name}'s corresponding ELF could not be found in the following list {orig_files}")
								continue
							identifier = "/".join([repo_path, directory, file_name])
							file_names_dictionary[identifier] = {"repo_path": repo_path,
																	"assembly_file": file_name,
																	"ELF_file": ELF, 
																	"assembly_sha": file2sha[file_name],
																	"ELF_sha": file2sha[ELF]
																	}
	return file_names_dictionary

def read_from_file(file_path: str) -> str: 
	'''
	Wrapper for f.read()
	'''
	with open(file_path, "r") as f: 
		file_contents = f.read()
	return file_contents

def functions_and_assembly(compile_path: str, file_names_dict): 
	'''
	Driver for the chunk.function_names function
	Returns the functions in the assembly file as well as the assembly read in as a string

	compile_path: path to the folder containin the partially and fully compiled files (i.e. assembly and ELF executable files)
	file_names_dict: dictionary returned from the collect_file_names function. It contains the repo path as well as the file names and hashes for each assembly and ELF
	'''
	repo_path = os.path.join(compile_path, file_names_dict["repo_path"])

	assembly_string_path = os.path.join(repo_path, file_names_dict["assembly_sha"])
	try: 
		assembly_string = read_from_file(assembly_string_path)
		ELF_path = os.path.join(repo_path, file_names_dict["ELF_sha"])
		fun_list = function_names(ELF_path, unopt_assembly_string)
	except: 
		print(f"there was an error with reading assembly file: {file_names_dict['repo_path'] +' file: '+ file_names_dict['assembly_file']}")
		return None, None

	return fun_list, assembly_string



def data_to_csv(out_file_name: str, unopt_compile_path: str, opt_compile_path: str, unopt_db: str, opt_db: str, check_for_duplicates = True) -> None: 
	if check_for_duplicates: 
		running_unopt_sha_set = set()

	# both dictionaries should have the same exact repo paths, assembly_file, and ELF_file values; however, the hashes will be different due to optimization levels
	# each key in the dictionary is the concatenation of the repo_path as well as the assembly file name 
	unoptimized_dictionary = collect_file_names(unopt_db)
	optimized_dictionary = collect_file_names(opt_db)
	for assembly_identifier in tqdm(unoptimized_dictionary):

		if check_for_duplicates: 
			# skip if the assembly hash has already been processed before
			if unoptimized_dictionary[assembly_identifier]["assembly_sha"] in running_unopt_sha_set: 
				continue
			else: 
				running_unopt_sha_set.add(unoptimized_dictionary[assembly_identifier]["assembly_sha"])

		if assembly_identifier in optimized_dictionary: 
			# get functions for both assembly files as well as corresponding assembly files
			unopt_fun_list, unopt_assembly_string = functions_and_assembly(unopt_compile_path, unoptimized_dictionary[assembly_identifier])
			opt_fun_list, opt_assembly_string = functions_and_assembly(opt_compile_path, optimized_dictionary[assembly_identifier])
			# ensure that the files were able to be red and that there is parity between functions
			if unopt_fun_list and opt_fun_list and set(unopt_fun_list) == set(opt_fun_list):
				# break out if the number of functions is above 300, a quick workaround
				if len(unopt_fun_list) > 300 or len(unopt_assembly_string) > 3000000:
					continue
				# dictionary where keys are function names and values are the assembly
				chunk_unopt_assembly = chunk_assembly(unopt_fun_list, unopt_assembly_string)
				chunk_opt_assembly = chunk_assembly(opt_fun_list, opt_assembly_string)
				for function_name in chunk_unopt_assembly: 
					#TODO: Add the repo path and the assembly hash so you can easily lookup the files for debugging
					csv_row = [assembly_identifier, function_name, chunk_unopt_assembly[function_name], chunk_opt_assembly[function_name]]
					write_to_csv(out_file_name, csv_row)
			else:
				print(f"the file {assembly_identifier} had inconsistencies in functions between the unopt and the opt versions\n\n \
							the set of unoptimized functions is {unopt_fun_list}\n \
							and the set of optimized functions is {opt_fun_list}\n\n")
		else: 
			print(f"the file {assembly_identifier} does not exist in the optimized dictionary")


