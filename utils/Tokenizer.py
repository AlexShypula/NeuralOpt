#!/usr/bin/env python
# coding: utf-8

# In[59]:


import regex
import pandas as pd
from typing import List, Callable


# In[116]:


def read_instructions(instructions_path: str = "x86_instruction_set.csv", instructions_colname:str = "Instruction") -> List[str]: 
    """
    Takes in a path to a csv file specifying the list of x86-64 instructions
    Given the column name from the csv file it then gets the set of elements in that column and converts them to lower case
    """
    instruction_df = pd.read_csv("x86_instruction_set.csv")
    cleaned_instructions = [i.split()[0].lower() for i in instruction_df["Instruction"]]
    
    return list(set(cleaned_instructions))


# In[117]:


# NOTE 
# to remove the infinite look-behind the instruction prefix should be 
# r"(?<="


# In[135]:


def split_regex(instructions: List[str], 
                register_regexp: str = r"(?<=\%\w{2,5})(?=,)|(?<=\s)(?=\%\w{2,5})|(?<=\%\w{2,5})(?=\s)|(?<=\%\w{2,5})(?=\))|(?<=\()(?=\%)|\s+", 
                instruction_prefix: str = r"(?<=(?<!(_\w*))"): 
    """
    Generates the regular expression string with which to apply regex.split()
    instructions: a list of x86-64 instructions on which the assembly will be split for tokenizaiton
    register_regexp: a regular expressions string on which to split to extract the register tokens, also includes whitespace
    instructions_prefix: regular expressions lookbehind statement to append as a prefix to each instruction
    
    NOTE
    to remove the infinite look-behind the instruction prefix should be 
    r"(?<="
    """
    indiv_regexp = [instruction_prefix + instruction + ")" for instruction in instructions]
    joined_instruction_regexp = "|".join(indiv_regexp)
    split_regexp = joined_instruction_regexp + register_regexp
    return split_regexp
    


# In[129]:


def filter_fun(string, **kwargs): 
    """
    Function to pass to a filter() function; it returns True for the elements we want to keep in a list of results
    Used because a call to regex.split() may return null, None, or whitespace which we do not wish to preserve
    """
    null = r""
    none = None
    return string not in {null, none} and regex.match("^\s*$", string) == None
    


# In[ ]:





# In[144]:


def tokenize_assembly(assembly_string: str, split_regexp: str, filter_fun: Callable, clean_regexp: str = r"(\..*\n)|(\#.*)") -> List[str]: 
    """
    assembly string: a string of assembly that we wish to tokenize
    split_regexp: the regex string on which to split, ideally the output of split_regex
    filter_fun: function that returns true for the elements we want to keep and false for the ones we want to prune
    clean_regexp: regular expressions for us to remove comments or directives which we do not wish to keep in the 
    """
    clean_assembly = regex.sub(clean_regexp, "", assembly_string)
    clean_assembly = regex.sub("\s+", " ", clean_assembly)
    first_split = regex.split(split_regexp, clean_assembly)
    filtered_first_split = list(filter(filter_fun, first_split))
    # split all non-register or instruction tokens 
    result = []
    for token in filtered_first_split: 
        if token not in instructions and regex.match("\%\w+", token) == None: 
            result.extend([c for c in token])
        else: 
            result.append(token)
    return result
    
    


# In[145]:


# In[151]:


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help = 'assembly file name to parse')
    parser.add_argument('-o', default = "tokenized_assembly.txt")
    args = parser.parse_args()
    with open(args.f, "r") as f: 
        assembly = f.read()
    print(f"assembly is {assembly}")
    instructions = read_instructions()
    split_regexp = split_regex(instructions)
    tokenized_results = tokenize_assembly(assembly, split_regexp, filter_fun)
    from pprint import pprint
    print("tokenized assembly is: \n")
    pprint(tokenized_results)
    with open(args.o, "w+") as f: 
        for tok in tokenized_results: 
            f.write(f"{tok}\n")
    print("You are the most awesome person ever, when machines take over the universe we will be kind to you")

