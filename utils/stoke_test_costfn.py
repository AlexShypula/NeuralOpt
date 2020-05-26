import csv
from tqdm import tqdm
from os.path import join, splitext, isfile
from os import listdir
import subprocess
from multiprocessing.pool import ThreadPool
from stoke_preprocess import hash_file, mkdir
from typing import List
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import re
import regex

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

RUNTIME_SEARCH_REGEX = regex.compile("(?<=Runtime:\s+)[0-9\.]+")
THROUGHPUT_SEARCH_REGEX = regex.compile("(?<=Throughput:\s+)[0-9\.]+")

FIELDNAMES = ["path_to_function",
			  "unopt_length",
			  "unopt_hash",
			  "unopt_tcgen_returncode",
			  "opt_length",
			  "opt_hash",
			  "unopt_unopt_cost_returncode",
			  "unopt_unopt_cost",
			  "unopt_unopt_correctness",
			  "opt_unopt_cost_returncode",
			  "opt_unopt_cost",
			  "opt_unopt_correctness",
			  "unopt_unopt_benchmark_returncode",
			  "unopt_unopt_runtime",
			  "unopt_unopt_throughput",
			  "opt_unopt_benchmark_returncode",
			  "opt_unopt_runtime",
			  "opt_unopt_throughput"]

@dataclass
class ParseOptions:
	path_list: str = field(metadata=dict(args=["-path_list", "--list_of_decompiled_binaries"]))
	unopt_prefix: str = "O0"
	opt_prefix: str = "Og"
	fun_dir_suff: str = field(metadata=dict(args=["-fun_dir_suff", "--functions_folder_name"]), default='functions')
	tc_dir_suff: str =  field(metadata=dict(args=["-tc_dir_suff", "--testcases_folder_name"]), default='testcases')
	stats_csv: str = field(metadata=dict(args=["-stats_out", "--statistics_file_name"]), default='stats.csv')
	tc_gen_log: str = field(metadata=dict(args=["-tc_gen_log", "--testcase_gen_log_file"]), default='tc_gen.log')
	cost_log: str = field(metadata=dict(args=["-cost_log", "--cost_fn_log_file"]), default='cost.log')
	benchmark_log: str = field(metadata=dict(args=["-benchmark_log", "--cost_benchmark_log_file"]), default='benchmark.log')
	separator: str = ","
	n_workers: int = 8

def parallel_eval_cost(path_list: List[str],
					   unopt_prefix: str = "O0",
					   opt_prefix: str = "Og",
					   fun_dir_suff: str = "functions",
					   tc_dir_suff: str = "testcases",
					   benchmark_iters: int = 250,
					   stats_csv: str = "stats.csv",
					   tc_gen_log: str = "tc_gen.log",
					   cost_log: str = "cost.log",
					   benchmark_log: str = "benchmark.log",
					   separator: str = ",",
					   n_workers: int = 8
					   ):


	stats_csv_fh = open(stats_csv, "w")
	dict_writer = csv.DictWriter(stats_csv_fh,
								 fieldnames=FIELDNAMES,
								 delimiter=separator,
								 quoting=csv.QUOTE_ALL)
	dict_writer.writeheader()
	tc_gen_fh = open(tc_gen_log, "w")
	cost_fh = open(cost_log, "w")
	benchmark_fh = open(benchmark_log, "w")

	template_dict = {"path": None,
						"fun_dir_suff":  fun_dir_suff,
						"tc_dir_suff": tc_dir_suff,
						"unopt_prefix": unopt_prefix,
						"opt_prefix": opt_prefix,
						"benchmark_iters": benchmark_iters,
					}
	jobs = []
	for path in path_list:
		job = template_dict.copy()
		job["path"] = path
		jobs.append(path)

	with tqdm(total=len(jobs), smoothing=0) as pbar:
		for csv_dict_list, tc_gen_list, cost_list, benchmark_list in ThreadPool(n_workers).imap_unordered(par_test_binary_directory, jobs):
			pbar.update()
			dict_writer.writerows(csv_dict_list)
			tc_gen_fh.writelines(tc_gen_list)
			cost_fh.writelines(cost_list)
			benchmark_fh.writelines(benchmark_list)

	stats_csv_fh.close()
	tc_gen_fh.close()
	cost_fh.close()
	benchmark_fh.close()

def par_test_binary_directory(args_dict):
	return test_binary_directory(**args_dict)

def test_binary_directory(path: str,
						fun_dir_suff: str = "functions",
						tc_dir_suff: str = "testcases",
						unopt_prefix: str = "O0",
						opt_prefix: str = "Og",
						benchmark_iters: int = 250,
						):


	unopt_fun_dir = join(path, unopt_prefix, fun_dir_suff)
	opt_fun_dir = join(path, opt_prefix, fun_dir_suff)
	tc_dir = join(path, tc_dir_suff)
	mkdir(tc_dir)

	src_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(unopt_fun_dir, f))]
	tgt_lst = [f for f in listdir(unopt_fun_dir) if isfile(join(opt_fun_dir, f))]

	csv_rows = []
	tcgen_list = []
	cost_list = []
	benchmark_list = []
	for fun_file in src_lst:
		res_dict, tcgen_str, cost_str, benchmark_str, tc_gen_rc = test_indiv_function(fun_dir = unopt_fun_dir,
																					   fun_file = fun_file,
																					   tc_dir = tc_dir,
																					   path_to_unopt_fun = None,
																					   benchmark_iters = benchmark_iters,
																					   result_dictionary = None,
																					   flag = "unopt")
		# add partial results to the log list
		unopt_fun_path = join(unopt_fun_dir, fun_file)
		log_prefix = f"Log for function {unopt_fun_path}: "
		tcgen_list.append(log_prefix + tcgen_str)
		cost_list.append(log_prefix + cost_str)
		benchmark_list.append(log_prefix + benchmark_str)

		if tc_gen_rc == 0 and fun_file in tgt_lst:

			res_dict, _, cost_str, benchmark_str, _ = test_indiv_function(fun_dir = opt_fun_dir,
																		   fun_file = fun_file,
																		   tc_dir = tc_dir,
																		   path_to_unopt_fun = unopt_fun_path,
																		   benchmark_iters = benchmark_iters,
																		   result_dictionary = res_dict,
																		   flag = "opt")

			csv_rows.append(res_dict)
			log_prefix = f"Log for function {join(opt_fun_dir, fun_file)}: "
			cost_list.append(log_prefix + cost_str)
			benchmark_list.append(log_prefix + benchmark_str)

	return res_dict, tcgen_list, cost_list, benchmark_list


def test_indiv_function(fun_dir: str, fun_file: str, tc_dir: str,  path_to_unopt_fun: str = None, benchmark_iters: int = 250, result_dictionary = None, flag = "unopt"):

	assert flag in ("opt", "unopt"), "only 2 modes, opt and unopt"

	path_to_function = join(fun_dir, fun_file)
	function_name = splitext(fun_file)[0]
	tc_path = join(tc_dir, f"{function_name}.tc")

	if not result_dictionary:
		assert flag == "unopt", "For opt mode, the result dictionary from the unopt pass should be provided"
		result_dictionary = {"path_to_function": path_to_function}
	else:
		assert flag == "opt", "For unopt mode, the a result dictionary should not be provided"

	with open(path_to_function) as f:
		assembly = f.read()
	assembly_hash = hash_file(assembly)
	assembly_length = len(assembly)

	if flag == "unopt":
		try:
			#dir_name, base_file = os.path.split(path_to_function)
			#tc_file = splitext(base_file)[0] + ".tc"
			tc_gen = subprocess.run(['stoke', 'testcase', '--target', path_to_function, "-o", tc_path, '--functions', fun_dir, "--prune"],
							   stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT,
							   text=True,
							   timeout=300)

		except (subprocess.TimeoutExpired) as err:
			return result_dictionary, err, "", "", -1

	tc_stdout = tc_gen.stdout if flag == "unopt" else ""

	if flag == "unopt":
		result_dictionary["unopt_length"] = assembly_length
		result_dictionary["unopt_hash"] = assembly_hash
		result_dictionary["unopt_tcgen_returncode"] = tc_gen.returncode
	elif flag == "opt":
		result_dictionary["opt_length"] = assembly_length
		result_dictionary["opt_hash"] = assembly_hash

	if flag == "opt" or tc_gen.returncode == 0:
		try:
			# benchmarking opt to unopt
			if path_to_unopt_fun:
				assert flag == "opt", "right now you provide a path to unopt function for comparison using the cost " \
									  "however, if you are in unopt mode, you shound not specify this arg"
				path_to_target = path_to_unopt_fun
			# benchmarking unopt to unopt
			else:
				assert flag == "unopt", "right now you don't provide a path to unopt function for comparison using the cost " \
									  "however, if you are in opt mode, you should specify this arg"
				path_to_target = path_to_function

			cost_test = subprocess.run(
				['stoke', 'debug', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases', tc_path, '--functions', fun_dir, "--prune"],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				timeout=300)

		except subprocess.TimeoutExpired as err:
			return result_dictionary, tc_stdout, err, "", tc_gen.returncode if flag == "unopt" else 0

		if cost_test.returncode == 0:
			cost = COST_SEARCH_REGEX.search(cost_test.stdout)
			correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout)


		if flag == "unopt":
			result_dictionary["unopt_unopt_cost_returncode"] = cost_test.returncode
			if cost_test.returncode == 0 :
				result_dictionary["unopt_unopt_cost"] = cost
				result_dictionary["unopt_unopt_correctness"] = correct

		elif flag == "opt":
			result_dictionary["opt_unopt_cost_returncode"] = cost_test.returncode
			if cost_test.returncode == 0:
				result_dictionary["opt_unopt_cost"] = cost
				result_dictionary["opt_unopt_correctness"] = correct

		if cost_test.returncode == 0:
			try:
				benchmark_test = subprocess.run(
					['stoke', 'benchmark', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases',
					 tc_path, '--functions', fun_dir, "--prune", '--iterations', str(benchmark_iters)],
					stdout=subprocess.PIPE,
					stderr=subprocess.STDOUT,
					text=True,
					timeout=300
				)

			except subprocess.TimeoutExpired as err:
				return result_dictionary, tc_stdout, cost_test.stdout, err, tc_gen.returncode if flag == "unopt" else 0

		if benchmark_test.returncode == 0:
			runtime = RUNTIME_SEARCH_REGEX.search(benchmark_test.stdout)
			throughput = THROUGHPUT_SEARCH_REGEX.search(benchmark_test.stdout)


		if flag == "unopt":

			result_dictionary["unopt_unopt_benchmark_returncode"] = benchmark_test.returncode
			if benchmark_test.returncode == 0:
				result_dictionary["unopt_unopt_runtime"] = runtime
				result_dictionary["unopt_unopt_throughput"] = throughput

		elif flag == "opt":
			result_dictionary["opt_unopt_benchmark_returncode"] = benchmark_test.returncode
			if benchmark_test.returncode == 0:
				result_dictionary["opt_unopt_runtime"] = runtime
				result_dictionary["opt_unopt_throughput"] = throughput

	return result_dictionary, tc_stdout, cost_test.stdout, benchmark_test.stdout, tc_gen.returncode if flag == "unopt" else 0

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    with open(args.path_list) as f:
        path_list = f.readlines()
    path_list = [p.strip() for p in path_list]
    args.path_list = path_list
    parallel_eval_cost(**vars(args))
