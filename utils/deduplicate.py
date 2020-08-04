from stoke_preprocess import hash_file
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from tqdm import tqdm

@dataclass
class ParseOptions:
    source: str = field(metadata=dict(args=["-src", "--source"]))
    target: str = field(metadata=dict(args=["-tgt", "--target"]))
    source_rewrite: str = field(metadata=dict(args=["-src_out", "--source_output"]))
    target_rewrite: str = field(metadata=dict(args=["-tgt_out", "--target_output"]))

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    sha_set = set()
    src_final = open(args.source_rewrite, "w")
    tgt_final = open(args.target_rewrite, "w")
    dups = 0
    with open(args.source) as src, open(args.target) as tgt:
        src_lines = src.readlines()
        tgt_lines = tgt.readlines()
        pbar = tqdm(total = len(src_lines))
        for s, t in zip(src_lines, tgt_lines):
            sha = hash_file(s.strip().lower())
            if sha not in sha_set:
                src_final.write(s + "\n")
                tgt_final.write(t + "\n")
                sha_set.add(sha)
            else:
                dups += 1
            pbar.update()
    print(f"total number of lines was {len(src_lines)} and dups were {dups}")
    src_final.close()
    tgt_final.close()
