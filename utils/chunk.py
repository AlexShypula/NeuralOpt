import regex as re
import csv
import subprocess
from typing import Dict, List
import os.path
from tqdm import tqdm
import pdb


def clean_function_line(assembly_function: str) -> str:  
	return re.sub("^:\s+", "", assembly_function)

def chunk_assembly(functions: List[str], raw_assembly: str) -> Dict[str, str]: 
	'''
	Chunks assembly (str format) based on the functions that exist in the assembly

	functions: list of functions on which to chunk the assembly (this comes from TOOD: Fill in, the ____ funciton which will pull the relevant functions from the assembly's correspingind ELF file)

	raw_assembly: assembly from a .s file that has been read in and is passed in a string

	Returns a dictionary "funciton2def" i.e. function to definition containing the funciton name as key and the corresponding assembly as the value

	'''
	function2def = {}
	function_regex = ["(?<=(\n)){}(?=:\n)".format(fun) for fun in functions]
	match_iterator = re.finditer("|".join(function_regex), raw_assembly)
	match = next(match_iterator)
	function = match.group()
	start = match.end(0)
	if len(functions) > 100:
		print(f"Len of Functions is {len(functions)}")
		t = tqdm(total = len(functions))
	for i, match in enumerate(match_iterator):
		if len(functions) > 100:
			print(f"iteration {i} start is {start} and end is {match.start(0)} len of file is {len(raw_assembly)}")
		if i == 858:
			pdb.set_trace()
		function2def[function] = clean_function_line(raw_assembly[start: match.start(0)])
		function = match.group(0)
		start = match.end(0)
		if len(functions) > 100:
			t.update()
	if len(functions) > 100:
		print(f"out of the match loop function, start is {start} and len of remainder is {len(raw_assembly[start:])}")
		pdb.set_trace()
	function2def[function] = clean_function_line(raw_assembly[start:])

	return function2def



def function_names(executable_filename: str, assembly_string: str) -> List[str]: 
	"""

	"""
	process1 = subprocess.Popen(["nm", executable_filename], stdout=subprocess.PIPE)
	process2 = subprocess.Popen(["grep", " T "], stdin=process1.stdout, stdout=subprocess.PIPE)
	process3 = subprocess.Popen(["sed", "s/.*T //"], stdin=process2.stdout, stdout=subprocess.PIPE)
	functions = process3.communicate()[0].decode("utf-8").split()

	#functions_in_assembly = [fun for fun in functions if fun in assembly_string]
	functions_in_assembly = [fun for fun in functions if re.search("(?<=(\n)){}(?=:\n)".format(fun), assembly_string)]

	return functions_in_assembly

def write_to_csv(filename: str, write_args: List[str], headers = ['file_path', 'function', 'unoptimized', 'optimized'])->None: 
	if not os.path.exists(filename): 
		with open(filename, "a+") as f: 
			csv.writer(f).writerow(headers)
			csv.writer(f).writerow(write_args)
	else: 
		with open(filename, "a") as f: 
			csv.writer(f).writerow(write_args)


