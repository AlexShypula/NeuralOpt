import json
import subprocess
import os
from os.path import join
from stoke_preprocess import hash_file
from make_data import function_path_to_optimized_function, function_path_to_functions_folder
from typing import List, Dict, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


@dataclass
class ParseOptions:
    path_to_data_files_dir: str = field(metadata=dict(args=["-path_to_data_dir", "-path_to_data_dir"]))
    path_to_hash2metadata: str = field(metadata=dict(args=["-path_to_hash2metadata", "--path_to_hash2metadata"]))
    asm_path_prefix: str = field(metadata=dict(args=["-asm_path_prefix", "--asm_path_prefix"]))
    path_to_error_log: str = field(metadata=dict(args=["-path_to_error_log", "--path_to_error_log"]))
    in_file_prefix: str = field(metadata=dict(args=["-in_file_prefix", "--in_file_prefix"]), default = "val")
    out_file_prefix: str = field(metadata=dict(args=["-out_file_prefix", "--out_file_prefix"]),
                                 default = "val_verified")
    src_file_suffix: str = field(metadata=dict(args=["-src_file_suffix", "--src_file_suffix"]), default = "src")
    tgt_file_suffix: str = field(metadata=dict(args=["-tgt_file_suffix", "--tgt_file_suffix"]), default = "tgt")


def main(path_to_data_files_dir: str, path_to_hash2metadata: str, in_file_prefix: str, out_file_prefix: str,
         asm_path_prefix: str, src_file_suffix: str, tgt_file_suffix: str, path_to_error_log: str):
    hash2metadata = json.load(open(path_to_hash2metadata))
    n_verified = 0
    total_cts = len(open(join(path_to_data_files_dir, in_file_prefix + "." + src_file_suffix)).readlines())
    pbar = tqdm(total=total_cts)
    with open(join(path_to_data_files_dir, in_file_prefix + "." + src_file_suffix)) as src_in_fh, \
            open(join(in_file_prefix, tgt_file_suffix)) as tgt_in_fh,  \
            open(path_to_error_log, "w+") as err_log_fh,  \
            open(join(path_to_data_files_dir, out_file_prefix + "." + src_file_suffix), "w+") as out_src_fh, \
            open(join(path_to_data_files_dir, out_file_prefix + "." + tgt_file_suffix), "w+") as out_tgt_fh:

        for src_asm, tgt_asm in zip(src_in_fh, tgt_in_fh):
            asm_hash = hash_file(src_asm.strip())
            metadata = hash2metadata[asm_hash]
            target_f = join(asm_path_prefix, metadata["base_asbly_path"])
            rewrite_f = function_path_to_optimized_function(target_f, optimized_flag="Og")
            fun_dir = function_path_to_functions_folder(target_f)
            def_in = metadata["def_in"]
            live_out = metadata["live_out"]
            heap_out = metadata["heap_out"]
            costfn = metadata["costfn"]
            verifed_correct, verified_stdout = verify_and_parse(target_f=target_f,
                                                                rewrite_f=rewrite_f,
                                                                fun_dir=fun_dir,
                                                                def_in=def_in,
                                                                live_out=live_out,
                                                                heap_out=heap_out,
                                                                costfn=costfn,
                                                                bound=64,
                                                                machine_output_f="tmp.txt",
                                                                strategy="bounded")
            if verifed_correct:
                out_src_fh.write(src_asm)
                out_tgt_fh.write(tgt_asm)
                n_verified+=1
            else:
                err_log_fh.write("function {} didn't verify\n\n".format(metadata["name"]))
                err_log_fh.write("{}\n\n".format(verified_stdout))
            pbar.update()
            pbar.set_description("verifying all assembly progress, {} have verified".format(n_verified))
            print("a total of {} of {} verified for {:2f}% percent".format(n_verified, total_cts, n_verified/total_cts))
    os.remove("tmp.txt")


def verify_rewrite(target_f: str,
                rewrite_f: str,
                fun_dir: str,
                def_in: str,
                live_out: str,
                heap_out: bool,
                costfn: str,
                bound: int = 64,
                machine_output_f: str = "tmp.txt",
                strategy: str = "bounded") -> (int, str):
    try:
        if heap_out:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
                 '--functions', fun_dir,
                 "--prune", "--live_dangerously"
                 "--def_in", def_in
                 "--live_out", live_out
                 "--distance", "hamming",
                 "--misalign_penalty", 1
                 "--sig_penalty", 9999,
                 "--cost", costfn,
                 "--bound", bound
                 "--heap_out"],
                stdout = subprocess.PIPE,stderr = subprocess.STDOUT,
                text = True, timeout = 300)
        else:
            verify_test = subprocess.run(
                ['/home/stoke/stoke/bin/stoke', 'debug', 'verify',
                 '--target', target_f,
                 '--rewrite', rewrite_f,
                 '--machine_output', machine_output_f,
                 '--strategy', strategy,
                 '--functions', fun_dir,
                 "--prune", "--live_dangerously"
                 "--def_in", def_in
                 "--live_out", live_out
                 "--distance", "hamming",
                 "--misalign_penalty", 1
                 "--sig_penalty", 9999,
                 "--cost", costfn,
                 "--bound", bound],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=300)
        return verify_test.returncode, verify_test.stdout
    except subprocess.TimeoutExpired as err:
        return -1, f"verify timed out with error {err}"


def parse_verify_machine_output(machine_output_f: str) -> bool:
    with open(machine_output_f) as fh:
        machine_output_dict = json.load(fh)
    verified_correct = machine_output_dict["verified"]
    return verified_correct


def verify_and_parse(target_f: str,
                        rewrite_f: str,
                        fun_dir: str,
                        def_in: str,
                        live_out: str,
                        heap_out: bool,
                        costfn: str,
                        machine_output_f: str = "tmp.txt",
                        strategy: str = "bounded"):

    verify_returncode, verify_stdout =  verify_rewrite(target_f=target_f,
                                        rewrite_f=rewrite_f,
                                        fun_dir=fun_dir,
                                        def_in=def_in,
                                        live_out=live_out,
                                        heap_out=heap_out,
                                        costfn=costfn,
                                        machine_output_f=machine_output_f,
                                        strategy=strategy)
    if verify_returncode == 0:
        verified_correct = parse_verify_machine_output(machine_output_f)
    else:
        verified_correct = False

    return verified_correct, verify_stdout


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    main(**args)
