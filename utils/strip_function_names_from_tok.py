import re
from typing import List
import sentencepiece as spm
from stoke_preprocess import merge_registers, stitch_together, _spec_char_rep, NEW_LINE_PATTERN_REPLACE, UNDERSCORE_PATTERN_REPLACE
from stoke_test_costfn import FUNCTION_NAME_REGEX
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from tqdm import tqdm


@dataclass
class ParseOptions:
    path_to_in_file: str = field(metadata=dict(args=["-in_f", "--path_to_in_file"]))
    path_to_out_file: str = field(metadata=dict(args=["-out_f", "--path_to_out_file"]))
    path_to_spm_model: str = field(metadata=dict(args=["-spm_model", "--path_to_spm_model"]))


def strip_function_name(assembly_string: str):
    match = FUNCTION_NAME_REGEX.search(assembly_string)
    if match == None:
        print(assembly_string)
    else:
        function_name = match.group()
    assembly_string = re.sub(f"\.{function_name}:[^\n]*\n", "", assembly_string)
    assembly_string = re.sub(f"\.size\s+{function_name},\s+.-.*", "", assembly_string)
    return assembly_string, function_name


def strip_and_retokenize(assembly_string: str, sent_piece_model: spm.SentencePieceProcessor):
    assembly_string = stitch_together(assembly_string)
    assembly_string, function_name = strip_function_name(assembly_string)
    assembly_string = _spec_char_rep(assembly_string.strip())
    bpe_string = merge_registers(sent_piece_model.EncodeAsPieces(assembly_string.strip()))
    return bpe_string, function_name


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(args.path_to_spm_model)
    with open(args.path_to_in_file, "r") as in_f, open(args.path_to_out_file, "w") as out_f:
        pbar = tqdm(total = len(in_f.readlines()), smoothing=.1, )
        for line in in_f:
            bpe_string, _ = strip_and_retokenize(line.strip(), spm_model)
            out_string = " ".join(bpe_string)
            out_f.write(out_string + "\n")
            pbar.update()
