import csv
import subprocess
import sentencepiece as spm
import re
import regex
import datetime
from tqdm import tqdm
from os.path import join, splitext, isfile
from os import listdir
from multiprocessing.pool import ThreadPool
from stoke_preprocess import hash_file, mkdir, process_raw_assembly, merge_registers, stitch_together, FINDALL_FUNCTIONS_PATTERN
from typing import List, Tuple
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from time import time, sleep

COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

RUNTIME_SEARCH_REGEX = regex.compile("(?<=Runtime:\s+)[0-9\.]+")
THROUGHPUT_SEARCH_REGEX = regex.compile("(?<=Throughput:\s+)[0-9\.]+")

UNSUPPORTED_REGEX = re.compile("(call\w{0,1}|j\w{1,4}) (\%\w+)")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")

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
			  "unopt_unopt_benchmark_time",
			  "opt_time",
			  "opt_overhead",
			  "opt_unopt_cost_time",
			  "opt_unopt_benchmark_time"]

STDOUTFIELDS = ["tcgen_str",
				"unopt_cost_str",
				"unopt_benchmark_str",
				"opt_cost_str",
				"opt_benchmark_str"]

ASBLYFIELDS = ["unopt_assembly",
				 "opt_assembly"]

BPETESTFIELDS = ["header_footer",
"orig_assembly",
"unopt_tunit_rc",
"unopt_bpe_tunit_stdout",
"unopt_tunit_assembly", 
"opt_tunit_rc",
"opt_bpe_tunit_stdout",
"opt_tunit_assembly",
"unopt_orig_unopt_bpe_cost_rc",
"unopt_orig_unopt_bpe_cost_stdout",
"unopt_orig_unopt_bpe_cost",
"unopt_orig_unopt_bpe_correctness",
"opt_orig_unopt_bpe_cost_rc",
"opt_orig_unopt_bpe_cost_stdout",
"opt_orig_unopt_bpe_cost",
"opt_orig_unopt_bpe_correctness"
]

TUNITORIGFIELDS = ["orig_tunit_rc",
"orig_tunit_stdout",
"orig_tunit_assembly",
"unopt_tunit_unopt_bpe_cost_rc",
"unopt_tunit_unopt_bpe_cost_stdout",
"unopt_tunit_unopt_bpe_cost",
"unopt_tunit_unopt_bpe_correctness",
"opt_tunit_unopt_bpe_cost_rc",
"opt_tunit_unopt_bpe_cost_stdout",
"opt_tunit_unopt_bpe_cost",
"opt_tunit_unopt_bpe_correctness"
]

SPMMODELFIELDS = ["unopt_full_canon_hash", "opt_full_canon_hash"]

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
	max_seq_len: int = field(metadata=dict(args=["-max_len", "--max_seq_len"]), default=None)
	spm_model_path: str = field(metadata=dict(args=["-spm_model_path", "--sent_piece_model_path"]), default=None)
	separator: str = ","
	n_workers: int = 8
	time: bool = field(metadata=dict(args=["-time", "--time_subprocesses"]), default = False)
	stdout_to_csv: bool = field(metadata=dict(args=["-stdout_to_csv", "--subprocess_out_to_csv"]), default = False)
	write_asbly: bool = field(metadata=dict(args=["-write_asbly", "--write_asseembly_to_csv"]), default = False)
	live_dangerously: bool = field(metadata=dict(args=["-live_dangerously", "--live_dangerously"]), default = False)
	filter_unsupported: bool = field(metadata=dict(args=["-filter_unsupported", "--filter_unsupported"]), default=False)
	test_bpe: bool = field(metadata=dict(args=["-test_bpe"]), default=False)
	tunit_on_orig: bool = field(metadata=dict(args=["-orig_tunit", "--tunit_on_orig"]), default=False)
	suppress_log: bool = field(metadata=dict(args=["-suppress_log", "--suppress_log"]), default=False)
	write_unopt_success_only: bool = field(metadata=dict(args=["-success_only", "--write_success_only"]), default=False)


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
					   max_seq_len: int = None,
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
					   live_dangerously: bool = False,
					   filter_unsupported: bool = False,
					   test_bpe: bool = False,
					   tunit_on_orig: bool = False,
					   suppress_log: bool = False,
					   write_unopt_success_only: bool = False
					   ):

	stop_watch = StopWatch()
	stop_watch.start()

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
	if test_bpe:
		field_names.extend(BPETESTFIELDS)
	if tunit_on_orig:
		field_names.extend(TUNITORIGFIELDS)
	if spm_model_path:
		field_names.extend(SPMMODELFIELDS)
	print(field_names)
	dict_writer = csv.DictWriter(stats_csv_fh,
								 fieldnames=field_names,
								 delimiter=separator,
								 quoting=csv.QUOTE_ALL)
	dict_writer.writeheader()
	tc_gen_fh = open(tc_gen_log, "w")
	cost_fh = open(cost_log, "w")
	benchmark_fh = open(benchmark_log, "w")

	asbly_hash_set = set()

	sent_piece = None
	if spm_model_path:
		sent_piece = spm.SentencePieceProcessor()
		sent_piece.Load(spm_model_path)

	template_dict = {"path": None,
						"asbly_hash_set": asbly_hash_set,
						"fun_dir_suff":  fun_dir_suff,
						"tc_dir_suff": tc_dir_suff,
						"unopt_prefix": unopt_prefix,
						"opt_prefix": opt_prefix,
						"benchmark_iters": benchmark_iters,
					    "max_testcases": max_testcases,
					 	"max_seq_len": max_seq_len,
					 	"time": time,
						"stdout_to_csv": stdout_to_csv,
					 	"spm_model": sent_piece,
					 	"write_asbly": write_asbly,
					 	"live_dangerously": live_dangerously,
					 	"filter_unsupported": filter_unsupported,
					 	"test_bpe": test_bpe,
					 	"tunit_on_orig": tunit_on_orig,
					 	"suppress_log": suppress_log,
					 	"write_unopt_success_only": write_unopt_success_only,
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
	stop_watch.stop()
	print(f"time it took to run the entire program was {datetime.timedelta(seconds = stop_watch.time)}")

def par_test_binary_directory(args_dict):
	return test_binary_directory(**args_dict)

def test_binary_directory(path: str,
						asbly_hash_set: List[str],
						fun_dir_suff: str = "functions",
						tc_dir_suff: str = "testcases",
						unopt_prefix: str = "O0",
						opt_prefix: str = "Og",
						benchmark_iters: int = 250,
						max_testcases: int = 1024,
						max_seq_len: int = None,
						time: bool = False,
						stdout_to_csv: bool = False,
						spm_model = None,
						write_asbly: bool = False,
						live_dangerously: bool = False,
						filter_unsupported: bool = False,
					    test_bpe: bool = False,
					    tunit_on_orig: bool = False,
						suppress_log: bool = False,
						write_unopt_success_only: bool = False
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
		res_dict, tcgen_str, cost_str, benchmark_str, tc_gen_rc, assembly_hash = test_indiv_function(fun_dir = unopt_fun_dir,
																					   fun_file = fun_file,
																					   tc_dir = tc_dir,
																					   asbly_hash_set = asbly_hash_set,
																					   path_to_unopt_fun = None,
																					   benchmark_iters = benchmark_iters,
																					   max_testcases = max_testcases,
																					   max_seq_len = max_seq_len,
																					   result_dictionary = None,
																					   flag = "unopt",
																					   time = time,
																					   spm_model = spm_model,
																					   write_asbly = write_asbly,
																					   live_dangerously = live_dangerously,
																					   filter_unsupported = filter_unsupported)

		asbly_hash_set.add(assembly_hash)
		# add partial results to the log list
		unopt_fun_path = join(unopt_fun_dir, fun_file)
		log_prefix = f"Log for function {unopt_fun_path}: "

		if not suppress_log:
			tcgen_list.append(log_prefix + tcgen_str)
			cost_list.append(log_prefix + cost_str)
			benchmark_list.append(log_prefix + benchmark_str)

		if stdout_to_csv:
			res_dict["tcgen_str"] = tcgen_str
			res_dict["unopt_cost_str"] = cost_str
			res_dict["unopt_benchmark_str"] = benchmark_str

		if tc_gen_rc == 0 and fun_file in tgt_lst:

			res_dict, _, cost_str, benchmark_str, _, _ = test_indiv_function(fun_dir = opt_fun_dir,
																		   fun_file = fun_file,
																		   tc_dir = tc_dir,
																		   asbly_hash_set = asbly_hash_set,
																		   path_to_unopt_fun = unopt_fun_path,
																		   benchmark_iters = benchmark_iters,
																		   max_seq_len = max_seq_len,
																		   result_dictionary = res_dict,
																		   flag = "opt",
																		   time = time,
																		   spm_model = spm_model,
																		   write_asbly = write_asbly,
																		   live_dangerously = live_dangerously,
																		   filter_unsupported = filter_unsupported)

			if not suppress_log:
				log_prefix = f"Log for function {join(opt_fun_dir, fun_file)}: "
				cost_list.append(log_prefix + cost_str)
				benchmark_list.append(log_prefix + benchmark_str)

			if test_bpe:
				res_dict = test_bpe_cost(fun_file = fun_file,
									 unopt_fun_dir = unopt_fun_dir,
									 opt_fun_dir = opt_fun_dir,
									 tc_dir = tc_dir,
									 result_dict = res_dict,
									 live_dangerously = live_dangerously,
									 tunit_orig = tunit_on_orig)

			if stdout_to_csv:
				res_dict["opt_cost_str"] = cost_str
				res_dict["opt_benchmark_str"] = benchmark_str
		if write_unopt_success_only:
			if res_dict.get("unopt_unopt_correctness") == "yes" and res_dict.get("opt_unopt_correctness") in ("yes", "no"):
				csv_rows.append(res_dict)
		else:
			csv_rows.append(res_dict)

	return csv_rows, tcgen_list, cost_list, benchmark_list


def test_indiv_function(fun_dir: str, fun_file: str, tc_dir: str,  asbly_hash_set: List[str],
						path_to_unopt_fun: str = None, benchmark_iters: int = 250, max_testcases: int = 1024,
						max_seq_len: int = None, result_dictionary = None, flag = "unopt", time = False,
						spm_model = None, write_asbly: bool = False, live_dangerously: bool = False,
						filter_unsupported: bool = False):

	assert flag in ("opt", "unopt"), "only 2 modes, opt and unopt"
	if max_seq_len:
		assert spm_model, "if using maximum sequence length, we need a sent-piece model to tokenize"

	if time:
		stop_watch = StopWatch()
		stop_watch.start()

	live_dangerously_str = "--live_dangerously" if live_dangerously else ""

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
	assembly_length = len(assembly)
	result_dictionary[f"{flag}_length"] = assembly_length

	if spm_model:
		asbly, _, function_name_list, _ = process_raw_assembly(raw_assembly=assembly, preserve_fun_names=True, preserve_semantics=True)
		full_canon_asbly = strip_function_names(asbly, function_name_list)
		assembly_hash = hash_file(full_canon_asbly) # used for deduplicating, following global convention
		tokenized_asbly = merge_registers(spm_model.EncodeAsPieces(asbly.strip()))
		result_dictionary[f"{flag}_bpe_len"] = len(tokenized_asbly)
		tokenized_asbly_string = " ".join(tokenized_asbly)
		bpe_hash = hash_file(tokenized_asbly_string) # has other purposes (i.e. use in training)
		result_dictionary[f"{flag}_hash"] = bpe_hash
		result_dictionary[f"{flag}_full_canon_hash"] = assembly_hash
		if write_asbly:
			result_dictionary[f"{flag}_assembly"] = " ".join(tokenized_asbly)
		if max_seq_len:
			if len(tokenized_asbly) > max_seq_len:
				return result_dictionary, "exceeds max seq len", "exceeds max seq len", "exceeds max seq len", -15213, assembly_hash
	else:
		assembly_hash = hash_file(assembly)
		result_dictionary[f"{flag}_hash"] = assembly_hash

	if assembly_hash in asbly_hash_set:
		return result_dictionary, "duplicate", "duplicate", "duplicate", -42, assembly_hash

	if filter_unsupported:
		match = UNSUPPORTED_REGEX.search(assembly)
		if match:
			return result_dictionary, f"unsupported operation: {match.group()}", f"unsupported operation: {match.group()}", \
			f"unsupported operation: {match.group()}", -15213, assembly_hash

	if flag == "unopt":
		try:
			#dir_name, base_file = os.path.split(path_to_function)
			#tc_file = splitext(base_file)[0] + ".tc"
			if time:
				stop_watch.new_event("tcgen")
				stop_watch.tcgen.start()

			tc_gen = subprocess.run(['stoke', 'testcase', '--target', path_to_function, "-o", tc_path,
									 '--functions', fun_dir, "--prune", '--max_testcases', str(max_testcases),
									 live_dangerously_str],
							   stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT,
							   text=True,
							   timeout=25)

			result_dictionary["unopt_tcgen_returncode"] = tc_gen.returncode

			if time:
				stop_watch.tcgen._stop_timing()
				result_dictionary["tcgen_time"] = stop_watch.tcgen.time

			if tc_gen.returncode != 0:
				if time:
					stop_watch.stop()
					result_dictionary[f"{flag}_time"] = stop_watch.time
					result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				return result_dictionary, tc_gen.stdout, "", "", tc_gen.returncode, assembly_hash

		except (subprocess.TimeoutExpired) as err:
			if time:
				stop_watch.stop()
				result_dictionary[f"{flag}_time"] = stop_watch.time
				result_dictionary[f"{flag}_overhead"] = stop_watch.overhead
				result_dictionary["tcgen_time"] = stop_watch.tcgen.time

			return result_dictionary, str(err), "", "", -1, assembly_hash

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
				['stoke', 'debug', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases',
				 tc_path, '--functions', fun_dir, "--prune", live_dangerously_str],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				timeout=25)

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

			return result_dictionary, tc_stdout, str(err), "", tc_gen.returncode if flag == "unopt" else 0, assembly_hash

		# if cost_test.returncode == 0:
		try:
			if time:
				stop_watch.new_event("benchmark_test")
				stop_watch.benchmark_test.start()

			benchmark_test = subprocess.run(
				['stoke', 'benchmark', 'cost', '--target', path_to_target, '--rewrite', path_to_function, '--testcases',
				 tc_path, '--functions', fun_dir, "--prune", '--iterations', str(benchmark_iters), live_dangerously_str],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				timeout=25
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

			return result_dictionary, tc_stdout, cost_test.stdout, str(err), tc_gen.returncode if flag == "unopt" else 0, assembly_hash

	if time:
		stop_watch.stop()
		result_dictionary[f"{flag}_time"] = stop_watch.time
		result_dictionary[f"{flag}_overhead"] = stop_watch.overhead

	return result_dictionary, tc_stdout, cost_test.stdout, benchmark_test.stdout, tc_gen.returncode if flag == "unopt" else 0, assembly_hash


def test_bpe_cost(fun_file: str, unopt_fun_dir: str, opt_fun_dir: str, tc_dir: str, result_dict, live_dangerously: bool = False, tunit_orig: bool  = False):
	orig_unopt_file = join(unopt_fun_dir, fun_file)

	unopt_bpe_dir = join(unopt_fun_dir, "bpe_tunit")
	opt_bpe_dir = join(opt_fun_dir, "bpe_tunit")
	mkdir(unopt_bpe_dir)
	mkdir(opt_bpe_dir)

	file_base = splitext(fun_file)[0]
	raw_file = file_base + ".raw"
	tunit_file = file_base + ".tunit"

	testcases_f = join(tc_dir, file_base + ".tc")

	raw_unopt_f = join(unopt_bpe_dir, raw_file)
	raw_opt_f = join(opt_bpe_dir, raw_file)
	unopt_tunit_f = join(unopt_bpe_dir, tunit_file)
	opt_tunit_f = join(opt_bpe_dir, tunit_file)

	unopt_string, header_footer = bpe2formatted(result_dict["unopt_assembly"], remove_footer = True)
	opt_string, _ = bpe2formatted(result_dict["opt_assembly"], header_footer, remove_footer = True)
	result_dict["header_footer"] = header_footer
	with open(raw_unopt_f, "w") as raw_unopt_fh, open(raw_opt_f, "w") as raw_opt_fh:
		raw_unopt_fh.write(unopt_string)
		raw_opt_fh.write(opt_string)

	unopt_tunit_rc, unopt_tunit_stdout = make_tunit_file(raw_unopt_f, unopt_tunit_f, unopt_fun_dir, live_dangerously)
	result_dict["unopt_tunit_rc"] = unopt_tunit_rc
	result_dict["unopt_bpe_tunit_stdout"] = unopt_tunit_stdout
	if unopt_tunit_rc != 0 :
		result_dict["unopt_tunit_assembly"] = unopt_string
		return result_dict

	opt_tunit_rc, opt_tunit_stdout = make_tunit_file(raw_opt_f, opt_tunit_f, unopt_fun_dir, live_dangerously)
	result_dict["opt_tunit_rc"] = opt_tunit_rc
	result_dict["opt_bpe_tunit_stdout"] = opt_tunit_stdout
	if opt_tunit_rc != 0 :
		result_dict["opt_tunit_assembly"] = opt_string
		return result_dict

	unopt_cost_rc, unopt_cost_stdout, unopt_cost, unopt_correct = test_costfn(target_f= orig_unopt_file,
																			  rewrite_f = unopt_tunit_f,
																			  testcases_f = testcases_f,
																			  fun_dir = unopt_fun_dir,
																			  live_dangerously = live_dangerously)

	opt_cost_rc, opt_cost_stdout, opt_cost, opt_correct = test_costfn(target_f=orig_unopt_file,
																			  rewrite_f=opt_tunit_f,
																			  testcases_f=testcases_f,
																			  fun_dir=unopt_fun_dir,
																			  live_dangerously=live_dangerously)


	with open(orig_unopt_file) as orig_fh, \
		open(unopt_tunit_f) as unopt_tunit_fh, \
		open(opt_tunit_f) as opt_tunit_fh:

		result_dict["orig_assembly"] = orig_fh.read()
		result_dict["unopt_tunit_assembly"] = unopt_tunit_fh.read()
		result_dict["opt_tunit_assembly"] = opt_tunit_fh.read()

	result_dict["unopt_orig_unopt_bpe_cost_rc"] = unopt_cost_rc
	result_dict["unopt_orig_unopt_bpe_cost_stdout"] = unopt_cost_stdout
	result_dict["unopt_orig_unopt_bpe_cost"] = unopt_cost
	result_dict["unopt_orig_unopt_bpe_correctness"] = unopt_correct

	result_dict["opt_orig_unopt_bpe_cost_rc"] = opt_cost_rc
	result_dict["opt_orig_unopt_bpe_cost_stdout"] = opt_cost_stdout
	result_dict["opt_orig_unopt_bpe_cost"] = opt_cost
	result_dict["opt_orig_unopt_bpe_correctness"] = opt_correct

	if tunit_orig:
		unopt_tunit_dir = join(unopt_fun_dir, "orig_tunit")
		mkdir(unopt_tunit_dir)
		orig_tunit_f = join(unopt_tunit_dir, fun_file + ".tunit")
		orig_tunit_rc, orig_tunit_stdout = make_tunit_file(orig_unopt_file, orig_tunit_f, unopt_fun_dir,live_dangerously)

		result_dict["orig_tunit_rc"] = orig_tunit_rc
		result_dict["orig_tunit_stdout"] = orig_tunit_stdout
		if orig_tunit_rc != 0:
			return result_dict

		unopt_tunit_cost_rc, unopt_tunit_cost_stdout, unopt_tunit_cost, unopt_tunit_correct = test_costfn(
			target_f=orig_tunit_f,
			rewrite_f=unopt_tunit_f,
			testcases_f=testcases_f,
			fun_dir=unopt_fun_dir,
			live_dangerously=live_dangerously)

		opt_tunit_cost_rc, opt_tunit_cost_stdout, opt_tunit_cost, opt_tunit_correct = test_costfn(
			target_f=orig_tunit_f,
			rewrite_f=opt_tunit_f,
			testcases_f=testcases_f,
			fun_dir=unopt_fun_dir,
			live_dangerously=live_dangerously)

		with open(orig_tunit_f) as orig_tunit_fh:
			result_dict["orig_tunit_assembly"] = orig_tunit_fh.read()

		result_dict["unopt_tunit_unopt_bpe_cost_rc"] = unopt_tunit_cost_rc
		result_dict["unopt_tunit_unopt_bpe_cost_stdout"] = unopt_tunit_cost_stdout
		result_dict["unopt_tunit_unopt_bpe_cost"] = unopt_tunit_cost
		result_dict["unopt_tunit_unopt_bpe_correctness"] = unopt_tunit_correct

		result_dict["opt_tunit_unopt_bpe_cost_rc"] = opt_tunit_cost_rc
		result_dict["opt_tunit_unopt_bpe_cost_stdout"] = opt_tunit_cost_stdout
		result_dict["opt_tunit_unopt_bpe_cost"] = opt_tunit_cost
		result_dict["opt_tunit_unopt_bpe_correctness"] = opt_tunit_correct

	return result_dict


def strip_function_names(assembly_text: str, function_name_list: List[str]):
	for function_name in function_name_list:
		assembly_text = re.sub(f"\.{function_name}:[^\n]*\n", "", assembly_text)
		assembly_text = re.sub(f"\.size\s+{function_name},\s+.-.*", "", assembly_text)
	return assembly_text


def create_header_footer(assembly_string: str):
    match = FUNCTION_NAME_REGEX.search(assembly_string)
    if match == None: 
        print(assembly_string)
    else: 
        function_name = match.group()
    header = f'''  .text\n  .global {function_name}\n  .type {function_name}, @function\n\n'''
    footer = f".size {function_name}, .-{function_name}"
    return header, footer


def bpe2formatted(assembly_string: str, header_footer : Tuple[str, str] = None, remove_footer: bool = False):
	# header is the zeroth indexed value in header_footer, and footer should be the fist indexed value
	un_bpe_string = stitch_together(assembly_string)
	if remove_footer:
		un_bpe_string = REMOVE_FOOTER_REGEX.sub("", un_bpe_string)
	if not header_footer:
		header_footer = create_header_footer(un_bpe_string)
	return header_footer[0] + un_bpe_string + header_footer[1], header_footer

def make_tunit_file(in_f: str, out_f: str, fun_dir: str, live_dangerously: bool = False):
	live_dangerously_str = "--live_dangerously" if live_dangerously else ""
	try:
		with open(out_f, "w") as f: 
			tunit_proc = subprocess.run(
				['stoke', 'debug', 'tunit', '--target', in_f,'--functions', fun_dir, "--prune", live_dangerously_str],
				stdout=f,
				stderr=subprocess.PIPE,
				text=True,
				timeout=25)

	except subprocess.TimeoutExpired as err:
		return -11747, err

	return tunit_proc.returncode, tunit_proc.stdout

def test_costfn(target_f: str, rewrite_f: str, testcases_f: str, fun_dir: str, live_dangerously: bool = False):
	live_dangerously_str = "--live_dangerously" if live_dangerously else ""
	try:
		cost_test = subprocess.run(
			['stoke', 'debug', 'cost', '--target', target_f, '--rewrite', rewrite_f, '--testcases',
			 testcases_f, '--functions', fun_dir, "--prune", live_dangerously_str, '--training_set', '{ 0 1 ... 31 }',
			 '--cost', '100*correctness+measured+latency', '--heap_out', '--stack_out'],
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			timeout=25)
		if cost_test.returncode == 0:
			cost = COST_SEARCH_REGEX.search(cost_test.stdout).group()
			correct = CORRECT_SEARCH_REGEX.search(cost_test.stdout).group()
		else:
			cost = -10701
			correct = "failed"
		return cost_test.returncode, cost_test.stdout, cost, correct

	except subprocess.TimeoutExpired as err:
		return -11785, err, -11785, "timeout"


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    with open(args.path_list) as f:
        path_list = f.readlines()
    path_list = [p.strip() for p in path_list]
    args.path_list = path_list
    parallel_eval_cost(**vars(args))
