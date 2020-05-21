import random
from tqdm import tqdm
from stoke_preprocess import hash_file
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser

random.seed(15213)

@dataclass
class ParseOptions:

    in_f_1: str = field(metadata=dict(args=["-i1", "--in_file_1"]))
    in_f_2: str = field(metadata=dict(args=["-i2", "--in_file_2"]))
    out_f_1: str = field(metadata=dict(args=["-o1", "--out_file_1"]))
    out_f_2: str = field(metadata=dict(args=["-o2", "--out_file_2"]))
    max_len: int = field(metadata=dict(args=["-max_len", "--max_line_length"]), default = 250)
    dedup: bool = field(metadata=dict(args=["-dedup", "--deduplicate"]), default = False)
    sample: int = field(metadata=dict(args=["-sample", "--sample_size"]), default = None)


def get_file_len(file_name: str):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i

def sample(in_f_1: str, in_f_2: str, out_f_1: str, out_f_2:str, sample: int):
    len_1 = get_file_len(in_f_1)
    len_2 = get_file_len(in_f_2)
    assert len_1 == len_2, f"Oh no, your file lengths are different, file 1 has {len_1} lines, file 2 has {len_2} lines"
    assert len_1 <= 100000, f"You have {len_1} lines in this file; are you sure you want to read this whole file in for sampling ?"

    indices = random.sample(range(len_1), sample)
    with open(in_f_1) as in_1, open(in_f_2) as in_2:
        f1 = in_1.readlines()
        f2 = in_2.readlines()
    with open(out_f_1, "w") as out_1, open(out_f_2, "w") as out_2:
        for i in tqdm(indices):
            out_1.write(f1[i])
            out_2.write(f2[i])

def filter_by_line_length(in_f_1: str, in_f_2: str, out_f_1: str, out_f_2:str, max_len = 250, delim = " ", dedup = False, **kwargs):
    len_1 = get_file_len(in_f_1)
    len_2 = get_file_len(in_f_2)
    assert len_1 == len_2, f"Oh no, your file lengths are different, file 1 has {len_1} lines, file 2 has {len_2} lines"


    out_1 = open(out_f_1, "w")
    out_2 = open(out_f_2, "w")
    if dedup:
        hashset = set()
    with tqdm(total=len_1, smoothing=0, position = 0, leave = True) as pbar:
        with open(in_f_1) as in_1, open(in_f_2) as in_2:
            for i, (line_1, line_2) in enumerate(zip(in_1, in_2)):
                pbar.update()
                if dedup:
                    line_1_hash = hash_file(line_1.strip())
                    line_2_hash = hash_file(line_2.strip())
                    if line_1_hash in hashset:
                        hashset.add(line_2_hash)
                        continue
                    elif line_2_hash in hashset:
                        hashset.add(line_1_hash)
                        continue
                    hashset.add(line_1_hash)
                    hashset.add(line_2_hash)
                if len(line_1.split(delim)) <= max_len and len(line_2.split(delim)) <= max_len:
                    out_1.write(line_1)
                    out_2.write(line_2)


if __name__ == "__main__":
    # point()
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    filter_by_line_length(**vars(args))
    if args.sample:
        sample(args.out_f_1, args.out_f_2, args.out_f_1, args.out_f_2, args.sample)