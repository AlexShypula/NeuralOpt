import subprocess
from tqdm import tqdm
from difflib import context_diff
from stoke_preprocess import stitch_together, mkdir
from sacrebleu import sentence_bleu
from os.path import join
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


@dataclass
class ParseOptions:
    src_file: str = field(metadata=dict(args=["-src_file", "--path_to_source_file"]))
    tgt_file: str = field(metadata=dict(args=["-tgt_file", "--path_to_target_file"]))
    hyp_file: str = field(metadata=dict(args=["-hyp_file", "--path_to_predictions"]))
    out_dir: str = field(metadata=dict(args=["-out_dir", "--desired_output_directory"]))


def bleu_and_diff(src_file: str, tgt_file: str, hyp_file: str, out_dir: str) -> None:

    with open (src_file) as f:
        src_lst = f.readlines()
    with open (tgt_file) as f:
        tgt_lst = f.readlines()
    with open (hyp_file) as f:
        hyp_lst = f.readlines()

    assert len(src_lst) == len(tgt_lst), "Oh no, your src_file number of lines is not equal to number of lines in tgt_file"
    assert len(tgt_lst) == len(hyp_lst), "Oh no, your src_file/tgt_file number of lines is not equal to number of lines in hyp_file"

    mkdir(out_dir)
    breakpoint()

    scores = [sentence_bleu([r.strip()], h.strip()) for r, h in zip(tgt_lst, hyp_lst)]

    z = zip(src_lst, tgt_lst, hyp_lst, scores)
    sorted_assemby = sorted(z, key=lambda zipped: zipped[3].score, reverse=True)

    src_lst, tgt_lst, hyp_lst, scores = zip(*sorted_assemby)

    for i in tqdm(range(len(src_lst))):
        source = stitch_together(src_lst[i].strip())
        reference = stitch_together(tgt_lst[i].strip())
        hypothesis = stitch_together(hyp_lst[i].strip())
        score_string = scores[i].format()

        diff = '\n'.join(context_diff(reference.split("\n"),
                          hypothesis.split("\n"),
                          fromfile='reference',
                          tofile="hypothesis"))


        with open(join(out_dir, f"{out_dir}_{str(i)}.txt"), "w") as out_f:

            out_f.write(f"Assembly Number {i} with bleu score {score_string}\n\n")
            out_f.write("-" * 40 + "\n")
            out_f.write("-" * 40 + "\n\n")

            out_f.write("The source (O0) assembly is: \n\n")
            out_f.write("-" * 40 + "\n\n")
            out_f.write(source)
            out_f.write("\n\n" + "-" * 40 + "\n")
            out_f.write("-" * 40 + "\n\n")

            out_f.write("The reference (Og) assembly is: \n\n"
                        ""
                        "")
            out_f.write("-" * 40 + "\n\n")
            out_f.write(reference)
            out_f.write("\n\n" + "-" * 40 + "\n")
            out_f.write("-" * 40 + "\n\n")

            out_f.write("The hypothesis (predicted) assembly is: \n\n")
            out_f.write("-" * 40 + "\n\n")
            out_f.write(hypothesis)
            out_f.write("\n\n" + "-" * 40 + "\n")
            out_f.write("-" * 40 + "\n\n")

            out_f.write("The diff is: \n\n")
            out_f.write("-" * 40 + "\n\n")
            out_f.write(diff)
            out_f.write("\n\n" + "-" * 40 + "\n")
            out_f.write("-" * 40 + "\n\n")


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()

    print("Now running diff on all")
    bleu_and_diff(src_file=args.src_file,
                  tgt_file=args.tgt_file,
                  hyp_file=args.hyp_file,
                  out_dir=args.out_dir)

    print("Now running compare-mt")
    subprocess.run(["compare-mt", args.tgt_file, args.hyp_file, "--output_directory", args.out_dir])

