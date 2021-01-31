import os
import re
from os import makedirs
from typing import List
from collections import OrderedDict


COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")

STOKE_TRAINING_SET_REGEX = re.compile("(?=})") # when you do sub, you need to include an extra space after the integer
TESTCASE_INDEX_FINDER=re.compile("(?<=Testcase )[0-9]+")


METADATA_SPLIT_PATTERN = re.compile("(?=# Text)")
FINDALL_FUNCTIONS_PATTERN = re.compile("(?<=.type ).*?(?=, @function)")
COMMENT_PATTERN = re.compile("#.*?(?=\n)")
WHITESPACE_PATTERN = re.compile("\n+")
FINDALL_LOCATIONS_PATTERN = re.compile("\..*?(?=:|\s)")

# should include new-lines and whitespace
FUNCTION_BEGIN_REGEX = re.compile("(?<=:)(\s+)")

HACK_TEXT = "  cmpq $0xffffff00, %rsp\n  je .continue\n  retq\n.continue:\n"


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


def _split_metadata(raw_assembly:str):
    metadata, assembly = METADATA_SPLIT_PATTERN.split(raw_assembly, maxsplit=1)
    return metadata, assembly


def process_raw_assembly(raw_assembly: str, preserve_fun_names: bool = True, preserve_semantics: bool = True):
    metadata, assembly = _split_metadata(raw_assembly)
    if preserve_fun_names:
        function_list = FINDALL_FUNCTIONS_PATTERN.findall(metadata)
    else:
        function_list = []
    assembly, orig2canon_loc_dict = _process_assembly(assembly, function_list, preserve_semantics)
    return metadata + assembly


def _process_assembly(assembly: str, function_list: List[str], preserve_semantics: bool):
    no_comments = COMMENT_PATTERN.sub("", assembly)
    no_extra_space = WHITESPACE_PATTERN.sub("\n", no_comments)
    clean_assembly, orig2canon_loc_dict = _canonicalize_labels(no_extra_space, function_list, preserve_semantics)
    return clean_assembly, orig2canon_loc_dict


def _canonicalize_labels(assembly: str, function_list: List[str], preserve_semantics: bool = True):
    raw_locs = FINDALL_LOCATIONS_PATTERN.findall(assembly)
    # make a list of the locations that we'll keep
    kept_locs = [".size"]
    for fun in function_list:
        kept_locs.append("."+fun)
        kept_locs.append(".-" + fun)
    # get all idiosyncratic locations to replace
    idiosyn_locs = [l for l in OrderedDict.fromkeys(raw_locs)
                               if l not in kept_locs]
    # canonicalized locations starting from 1
    if preserve_semantics:
        canon_locs = [".L"+ str(i+1) for i in range(len(idiosyn_locs))]
    else:
        canon_locs = [".LOC"] * len(idiosyn_locs)
    idiosyn2canon = {idiosyn: canon for idiosyn, canon in zip(idiosyn_locs, canon_locs)}
    for idiosyn, canon in idiosyn2canon.items():
        # replace all occurrences
        assembly = re.sub(idiosyn, canon, assembly)
    return assembly, idiosyn2canon

