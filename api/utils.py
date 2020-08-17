import os
import re
from os import makedirs


COST_SEARCH_REGEX = re.compile("(?<=Cost: )\d+")
CORRECT_SEARCH_REGEX = re.compile("(?<=Correct: )\w+")

FUNCTION_NAME_REGEX = re.compile("(?<=\.)[\w_]+(?=:)")
REMOVE_FOOTER_REGEX = re.compile(".size [\w_\s\-\.,]+")

STOKE_TRAINING_SET_REGEX = re.compile("(?=})") # when you do sub, you need to include an extra space after the integer

def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)


