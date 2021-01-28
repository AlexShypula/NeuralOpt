import os
import re
from os import makedirs


COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")

STOKE_TRAINING_SET_REGEX = re.compile("(?=})") # when you do sub, you need to include an extra space after the integer
TESTCASE_INDEX_FINDER=re.compile("(?<=Testcase )[0-9]+")

def _get_testcase_indices(testcase_string: str):
    testcase_indices = TESTCASE_INDEX_FINDER.findall(testcase_string)
    testcase_indices = [int(tc_idx) for tc_idx in set(testcase_indices)]
    return testcase_indices

def get_max_testcase_index(testcase_string: str):
    testcase_indices = _get_testcase_indices(testcase_string)
    return max(testcase_indices)

def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)

def function_path_to_optimized_function(path: str, optimized_flag: str = "Og"):
    split_path = path.split("/")
    # -1 -> function name, -2 -> "functions", -3 -> "O0/Og/..." flag
    split_path[-3] = optimized_flag
    return "/".join(split_path)


