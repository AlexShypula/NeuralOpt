import csv
import subprocess
import sentencepiece as spm
import re
import regex
from tqdm import tqdm
from os.path import join, splitext, isfile
from os import listdir
from multiprocessing.pool import ThreadPool
from stoke_preprocess import hash_file, mkdir, process_raw_assembly, merge_registers
from typing import List
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from time import time, sleep

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

RUNTIME_SEARCH_REGEX = regex.compile("(?<=Runtime:\s+)[0-9\.]+")
THROUGHPUT_SEARCH_REGEX = regex.compile("(?<=Throughput:\s+)[0-9\.]+")

FIELDNAMES = ["path_to_function",
			  "unopt_length",
			  "unopt_bpe_len",
			  "unopt_hash",
			  "unopt_tcgen_returncode",
			  "opt_length",
			  "opt_bpe_len",
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

TIMEFIELDS = ["unopt_time",
			  "unopt_overhead",
			  "tcgen_time",
			  "unopt_unopt_cost_time",
			  "unopt_unopt_benchamrk_time",
			  "opt_time",
			  "opt_overhead",
			  "opt_unopt_cost_time",
			  "opt_unopt_benchamrk_time"]

STDOUTFIELDS = ["tcgcn_str",
				"unopt_cost_str",
				"unopt_benchmark_str",
				"opt_cost_str",
				"opt_benchmark_str"]

ASBLYFIELDS = ["unopt_assembly",
				 "opt_assembly"]

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
	benchmark_iters: int = field(metadata=dict(args=["-benchmark_iters", "--benchmark_number_tests"]), default=250)
	max_testcases: int = field(metadata=dict(args=["-max_tc", "--max_testcases"]), default=1024)
	spm_model_path: str = field(metadata=dict(args=["-spm_model_path", "--sent_piece_model_path"]), default=None)
	separator: str = ","
	n_workers: int = 8
	time: bool = field(metadata=dict(args=["-time", "--time_subprocesses"]), default = False)
	stdout_to_csv: bool = field(metadata=dict(args=["-stdout_to_csv", "--subprocess_out_to_csv"]), default = False)
	write_asbly: bool = field(metadata=dict(args=["-write_asbly", "--write_asseembly_to_csv"]), default = False)


class StopWatch:
	def __init__(self):
		self.time = 0
		self.n_events = 0
		self.timing = False
	def start(self):
		assert self.timing == False
		self._start_time = time()
		self.timing = True
	def _stop_timing(self):
		assert self.timing == True
		self._end_time = time()
		self.time = self._end_time - self._start_time
		self.timing = False
	def new_event(self, name = None):
		self.n_events += 1
		if not name:
			name = f"unnamed_event_{self.n_events}"
		setattr(self, name, StopWatch())
	def _calculate_overhead(self):
		assert self.timing == False
		self.overhead = self.time
		for k,v in self.__dict__.items():
			#print(f"key: {k}, value: {v}")
			if type(v) == type(self):
				if v.timing == True:
					v._stop_timing()
				self.overhead -= v.time
		return self.overhead
	def stop(self):
		assert self.timing == True
		self._stop_timing()
		self._calculate_overhead()
		return self.time, self.overhead


def mini_watch_test():
	foo = StopWatch()
	foo.start()
	sleep(2)
	foo.new_event("bar")
	foo.bar.start()
	sleep(1)
	foo.bar.stop()
	foo.new_event("spam")
	foo.spam.start()
	sleep(2)
	foo.stop()
	print("foo: ", vars(foo))
	print("bar: ", vars(foo.bar))
	print("spam: ", vars(foo.spam))
	# foo should be ~5 total, ~2 overhead
	# bar should be ~1 total
	# spam should be ~2


def parallel_eval_cost(path_list: List[str],
					   unopt_prefix: str = "O0",
					   opt_prefix: str = "Og",
					   fun_dir_suff: str = "functions",
					   tc_dir_suff: str = "testcases",
					   benchmark_iters: int = 250,
					   max_testcases: int = 1024,
					   stats_csv: str = "stats.csv",
					   tc_gen_log: str = "tc_gen.log",
					   cost_log: str = "cost.log",
					   benchmark_log: str = "benchmark.log",
					   spm_model_path: str = None,
					   separator: str = ",",
					   n_workers: int = 8,
					   time: bool = False,
					   stdout_to_csv: bool = False,
					   write_asbly: bool = False,
					   ):


	if write_asbly:
		assert spm_model_path, "in order to collect the assembly, you need to specify a sentpiece model for processing"
	stats_csv_fh = open(stats_csv, "w")
	field_names = FIELDNAMES
	if time:
		field_names.extend(TIMEFIELDS)
	if stdout_to_csv:
		field_names.extend(STDOUTFIELDS)
	if write_asbly:
		field_names.extend(ASBLYFIELDS)
	dict_writer = csv.DictWriter(stats_csv_fh,
								 fieldnames=field_names,
								 delimiter=separator,
								 quoting=csv.QUOTE_ALL)
	dict_writer.writeheader()
	tc_gen_fh = open(tc_gen_log, "w")
	cost_fh = open(cost_log, "w")
	benchmark_fh = open(benchmark_log, "w")

	sent_piece = None
	if spm_model_path:
		sent_piece = spm.SentencePieceProcessor()
		sent_piece.Load(spm_model_path)

	template_dict = {"path": None,
						"fun_dir_suff":  fun_dir_suff,
						"tc_dir_suff": tc_dir_suff,
						"unopt_prefix": unopt_prefix,
						"opt_prefix": opt_prefix,
						"benchmark_iters": benchmark_iters,
					    "max_testcases": max_testcases,
					 	"time": time,
						"stdout_to_csv": stdout_to_csv,
					 	"spm_model": sent_piece,
					 	"write_asbly": write_asbly
					}
	jobs = []
	for path in path_list:
		job = template_dict.copy()
		job["path"] = path
		jobs.append(job)

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
						max_testcases: int = 1024,
						time: bool = False,
						stdout_to_csv: bool = False,
						spm_model = None,
						write_asbly: bool = False,
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
																					   max_testcases = max_testcases,
																					   result_dictionary = None,
																					   flag = "unopt",
																					   time = time,
																					   spm_model = spm_model,
																					   write_asbly = write_asbly)
		# add partial results to the log list
		unopt_fun_path = join(unopt_fun_dir, fun_file)
		log_prefix = f"Log for function {unopt_fun_path}: "

		tcgen_list.append(log_prefix + tcgen_str)
		cost_list.append(log_prefix + cost_str)
		benchmark_list.append(log_prefix + benchmark_str)

		if stdout_to_csv:
			res_dict["tcgen_str"] = tcgen_str
			res_dict["unopt_cost_str"] = cost_str
			res_dict["unopt_benchmark_str"] = benchmark_str

		if tc_gen_rc == 0 and fun_file in tgt_lst:

			res_dict, _, cost_str, benchmark_str, _ = test_indiv_function(fun_dir = opt_fun_dir,
																		   fun_file = fun_file,
																		   tc_dir = tc_dir,
																		   path_to_unopt_fun = unopt_fun_path,
																		   benchmark_iters = benchmark_iters,
																		   result_dictionary = res_dict,
																		   flag = "opt",
																		   time = time,
																		   spm_model = spm_model,
																		   write_asbly = write_asbly)

			log_prefix = f"Log for function {join(opt_fun_dir, fun_file)}: "
			cost_list.append(log_prefix + cost_str)
			benchmark_list.append(log_prefix + benchmark_str)
			if stdout_to_csv:
				res_dict["opt_cost_str"] = cost_str
				res_dict["opt_benchmark_str"] = benchmark_str

		csv_rows.append(res_dict)

	return csv_rows, tcgen_list, cost_list, benchmark_list


def test_indiv_function(fun_dir: str, fun_file: str, tc_dir: str,  path_to_unopt_fun: str = None,
						benchmark_iters: int = 250, max_testcases: int = 1024,
						result_dictionary = None, flag = "unopt", time = False, spm_model = None,
						write_asbly: bool = False):

	assert flag in ("opt", "unopt"), "only 2 modes, opt and unopt"

	if time:
		stop_watch = StopWatch()
		stop_watch.start()

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
	result_dictionary[f"{flag}_length"] = assembly_length
	result_dictionary[f"{flag}_hash"] = assembly_hash

	if spm_model:
		asbly = process_raw_assembly(raw_assembly=assembly, preserve_fun_names=True, preserve_semantics=True)
		tokenized_asbly = merge_registers(spm_model.EncodeAsPieces(asbly.strip()))
		result_dictionary[f"{flag}_bpe_len"] = len(tokenized_asbly)
		if write_asbly:
			result_dictionary[f"{flag}_assembly"] = " ".join(tokenized_asbly)

	if flag == "unopt":
		try:
			#dir_name, base_file = os.path.split(path_to_function)
			#tc_file = splitext(base_file)[0] + ".tc"
			if time:
				stop_watch.new_event("tcgen")
				stop_watch.tcgen.start()

			tc_gen = subprocess.run(['stoke', 'testcase', '--target', path_to_function, "-o", tc_path,
									 '--functions', fun_dir, "--prune", '--max_testcases', max_testcases],
							   stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT,
							   text=True,
							   timeout=300)

			result_dictionary["unopt_tcgen_returncode"] = tc_gen.returncode

			if time:
				stop_watch.tcgen._stop_timing()
				result_dictionary["tcgen_time"] = stop_watch.tcgen.time

			if tc_gen.returncode != 0:
				if time:
					stop_watch.stop()
					result_dictionary[f"{flag}_time"] = stop_watch.time
					result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				return result_dictionary, tc_gen.stdout, "", "", tc_gen.returncode

		except (subprocess.TimeoutExpired) as err:
			if time:
				stop_watch.stop()
				result_dictionary[f"{flag}_time"] = stop_watch.time
				result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				result_dictionary["tcgen_time"] = stop_watch.tcgen.time

			return result_dictionary, str(err), "", "", -1

	tc_stdout = tc_gen.stdout if flag == "unopt" else ""

	if flag == "opt" or tc_gen.returncode == 0:
		#TODO: remove this unnecessary if-statement as we can just execute it every-time it is called
		#if tc_gen returncode != 0 we have an early exit anyway.....
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

			if time:
				stop_watch.new_event("cost_test")
				stop_watch.cost_test.start()

			cost_test = subprocess.run(
				['stoke', 'debug', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases', tc_path, '--functions', fun_dir, "--prune"],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				timeout=300)

			if time:
				stop_watch.cost_test._stop_timing()
				result_dictionary[f"{flag}_unopt_cost_time"] = stop_watch.cost_test.time

			if cost_test.returncode == 0:
				cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
				correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()

			result_dictionary[f"{flag}_unopt_cost_returncode"] = cost_test.returncode
			if cost_test.returncode == 0:
				result_dictionary[f"{flag}_unopt_cost"] = cost
				result_dictionary[f"{flag}_unopt_correctness"] = correct

		except subprocess.TimeoutExpired as err:
			if time:
				stop_watch.stop()
				result_dictionary[f"{flag}_time"] = stop_watch.time
				result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				result_dictionary[f"{flag}_unopt_cost_time"] = stop_watch.cost_test.time

			return result_dictionary, tc_stdout, str(err), "", tc_gen.returncode if flag == "unopt" else 0

		# if cost_test.returncode == 0:
		try:
			if time:
				stop_watch.new_event("benchmark_test")

			benchmark_test = subprocess.run(
				['stoke', 'benchmark', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases',
				 tc_path, '--functions', fun_dir, "--prune", '--iterations', str(benchmark_iters)],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				timeout=300
			)

			if time:
				stop_watch.benchmark_test._stop_timing()
				result_dictionary[f"{flag}_unopt_benchmark_time"] = stop_watch.benchmark_test.time

			if benchmark_test.returncode == 0:
				runtime = RUNTIME_SEARCH_REGEX.search(benchmark_test.stdout).group()
				throughput = THROUGHPUT_SEARCH_REGEX.search(benchmark_test.stdout).group()

			result_dictionary[f"{flag}_unopt_benchmark_returncode"] = benchmark_test.returncode
			if benchmark_test.returncode == 0:
				result_dictionary[f"{flag}_unopt_runtime"] = runtime
				result_dictionary[f"{flag}_unopt_throughput"] = throughput

		except subprocess.TimeoutExpired as err:
			if time:
				stop_watch.stop()
				result_dictionary[f"{flag}_time"] = stop_watch.time
				result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				result_dictionary[f"{flag}_unopt_benchmark_time"] = stop_watch.benchmark_test.time

			return result_dictionary, tc_stdout, cost_test.stdout, str(err), tc_gen.returncode if flag == "unopt" else 0

	if time:
		stop_watch.stop()
		result_dictionary[f"{flag}_time"] = stop_watch.time
		result_dictionary[f"{flag}_overhead"] = stop_watch.overhead

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
