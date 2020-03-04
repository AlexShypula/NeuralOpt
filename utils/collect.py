import os
from pymongo import MongoClient
import utils
from tqdm import tqdm


def collect_file_names(database: str, collection: str = "repos") -> Dict[str, Dict]: 
	"""
	collects the file names the specified database (i.e. the assembly file, the corresponding ELF file, as well as the corresponding hashes for both)

	database: database to query for results

	returns a dictionary in which the key is the full path to the assembly file and its value is a dictionary with the repo_path, original assembly_file name, corresponding ELF file name, and their respective hashes
	"""

	client= MongoClient()
	db_compile_results = client[database][collection]

	file_names_dictionary = {}

	for compile_result in db_compile_results.find(): 
		if compile_result["num_binaries"] > 0: 
			orig_files = compile_result["binares"]
			sha256_files = compile_result['sha256']
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

					assembly_path = "/".join([repo_path, file_name])
					file_names_dictionary[assembly_path] = {"repo_path": repo_path,
															"assembly_file": file_name,
															"ELF_file": ELF, 
															"assembly_sha": file2sha[file_name],
															"ELF_sha": file2sha[ELF]
															}
	return file_names_dictionary

def read_from_file(file_path: str) -> str: 
	with open(file_path, "r") as f: 
		file_contents = f.read()
	return file_contents

def functions_and_assembly(compile_path, file_names_dict): 
	repo_path = os.path.join(compile_path, file_names_dict[repo_path])

	assembly_string_path = os.path.join(repo_path, file_names_dict["assembly_sha"])
	assembly_string = read_from_file(assembly_string_path)
	ELF_path = os.path.join(repo_path, file_names_dict["ELF_sha"])
	fun_list = utils.function_names(ELF_path, unopt_assembly_string)

	return fun_list, assembly_string



def data_to_csv(out_file_name: str, unopt_compile_path: str, opt_compile_path: str, unopt_db: str, opt_db: str, check_for_duplicates = True) -> None: 
	if check_for_duplicates: 
		running_unopt_sha_set = set()

	unoptimized_dictionary = collect_file_names(unopt_db)
	optimized_dictionary = collect_file_names(opt_db)

	for assembly_file_name in tqdm(unoptimized_dictionary):

		if check_for_duplicates: 
			if unoptimized_dictionary["assembly_sha"] in running_unopt_sha_set: 
				continue
			else: 
				running_unopt_sha_set.add(unoptimized_dictionary["assembly_sha"])

		if assembly_file_name in optimized_dictionary: 
			unopt_fun_list, unopt_assembly_string = functions_and_assembly(unopt_compile_path, unoptimized_dictionary[assembly_file_name])
			opt_fun_list, opt_assembly_string = functions_and_assembly(opt_compile_path, optimized_dictionary[assembly_file_name])

			if set(unopt_fun_names) == set(opt_fun_names): 
				# dictionary where keys are function names and values are the assembly 
				chunk_unopt_assembly = utils.chunk_assembly(unopt_fun_list, unopt_assembly_string)
				chunk_opt_assembly = utils.chunk_assembly(opt_fun_list, opt_assembly_string)
				for function_name in chunk_unopt_assembly: 
					csv_row = [assembly_file_name, function_name, chunk_unopt_assembly[function_name], chunk_opt_assembly[function_name]]
					utils.write_to_csv(out_file_name, csv_row)
			else:
				print(f"the file {assembly_file_name} had inconsistencies in functions between the unopt and the opt versions\n\n \
							the set of unoptimized functions is {set(unopt_fun_names)}\n \
							and the set of optimized funcitons is {set(opt_fun_names)}\n\n")

		else: 
			print(f"the file {assembly_file_name} does not exist in the optimized dictionary")


