import subprocess
import csv
from typing import List
import os
from os import makedirs
from os.path import splitext, join
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from stoke_preprocess import hash_file, process_raw_assembly
from stoke_test_costfn import test_costfn
import sentencepiece as spm


@dataclass
class ParseOptions:
    path_to_c_prog_list: str = field(metadata=dict(args=["-c_progs", "--path_to_c_prog_list"]))
    path_to_asbly_prefix_list: str = field(metadata=dict(args=["-asbly_names", "--path_to_asbly_prefix_list"]))
    destination_dir: str = field(metadata=dict(args=["-out_dir", "--out_directory"], default="processed_data"))


def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)

def merge_registers(a):
    '''
    merge back split registers
    '''
    res = []
    last = a[0]
    for current in a[1:]:
        if last == None:
            last = current
            continue
        if last != '▁%' and last != '%':
            res.append(last)
            last = current
        else:
            res.append(''.join([last, current]))
            last = None
    return res


def pre_process(c_progs: List[str], asbly_prefixes: List[str],
                destination_dir: str = "processed_data",
                csv_out: str = "cost_stats.csv",
                compilation_flags: List[str] = ["O0", "Og", "O2", "O3"]):

    stats_csv_fh = open(join(destination_dir, csv_out), "w")
    field_names = ["problem"]
    for compilation_flag in compilation_flags:
        field_names.append(f"{compilation_flag}-O0_cost")
        field_names.append(f"{compilation_flag}-O0_correct")
    dict_writer = csv.DictWriter(stats_csv_fh,
                                 fieldnames=field_names,
                                 delimiter=",",
                                 quoting=csv.QUOTE_ALL)
    dict_writer.writeheader()
    for i, (c_prog_pth, asbly_prefix) in enumerate(zip(c_progs, asbly_prefixes)):
        cost_dict = process_program(file_path=c_prog_pth,
                        assembly_file_name = asbly_prefix,
                        problem_no = i,
                        destination_dir=destination_dir,
                        compilation_flags = compilation_flags)
        dict_writer.writerow(cost_dict)

    stats_csv_fh.close()


def make_data(c_progs: List[str],
              asbly_prefixes: List[str],
              spm_model_path: str,
              destination_dir: str = "processed_data",
              data_dir = "model_data",
              src_flag = "O0",
              tgt_flag = "Og"
              ):
    model_data_dir = join(destination_dir, data_dir)
    mkdir(model_data_dir)
    train_src = open(join(model_data_dir, "train.src"), "w")
    train_tgt = open(join(model_data_dir, "train.tgt"), "w")
    val_src = open(join(model_data_dir, "val.src"), "w")
    val_tgt = open(join(model_data_dir, "val.tgt"), "w")
    test_src = open(join(model_data_dir, "test.src"), "w")
    test_tgt = open(join(model_data_dir, "test.tgt"), "w")

    sent_piece = spm.SentencePieceProcessor()
    sent_piece.Load(spm_model_path)

    for i, (c_prog_pth, asbly_prefix) in enumerate(zip(c_progs, asbly_prefixes)):
        program_directory = join(destination_dir, os.path.basename(os.path.dirname(c_prog_pth)))
        for compilation_flag in (src_flag, tgt_flag):
            function_dir = join(program_directory, compilation_flag, "functions")
            asbly_path = join(function_dir, asbly_prefix + ".s")
            with open(asbly_path, "r") as f:
                raw_asbly = f.read()
                processed_asbly, _, _, _ = process_raw_assembly(raw_asbly)
                tokenized_asbly = merge_registers(sent_piece.EncodeAsPieces(processed_asbly.strip()))
                asbly_str = " ".join(tokenized_asbly)

            if compilation_flag == "O0":
                if i < 15:
                    train_src.write(asbly_str+'\n')
                elif i < 20:
                    val_src.write(asbly_str+'\n')
                else:
                    test_src.write(asbly_str+'\n')
            else:
                if i < 15:
                    train_tgt.write(asbly_str+'\n')
                elif i < 20:
                    val_tgt.write(asbly_str+'\n')
                else:
                    test_tgt.write(asbly_str+'\n')

    train_src.close()
    train_tgt.close()
    val_src.close()
    val_tgt.close()
    test_src.close()


def process_program(file_path: str,
                    assembly_file_name: str,
                    problem_no: int,
                    destination_dir: str = "processed_data",
                    compilation_flags: List[str] = ["O0", "Og", "O2", "O3"]
                    ):
    assert compilation_flags[0] == "O0", "first compilation flag must be O0 for the purpose of making comparisons"
    #breakpoint()
    #file_prefix = splitext(file_path)[0]
    #program_directory = join(destination_dir, file_prefix)
    #model_data_dir = join(destination_dir, "model_data")
    problem_no = os.path.basename(os.path.dirname(file_path))
    program_directory = join(destination_dir, problem_no)
    testcases_directory = join(program_directory, "testcases")
    ### make directories
    mkdir(testcases_directory)
    cost_dict = {"problem": problem_no}
    for compilation_flag in compilation_flags:
        for subfolder in ("functions", "bin"):
            mkdir(join(program_directory, compilation_flag, subfolder))
        fxn_file = join(program_directory, compilation_flag, "bin", "fxn.o")
        bin_file = join(program_directory, compilation_flag, "bin", "a.out")
        fxn_compile = subprocess.run(["g++", "-std=c++11",f"-{compilation_flag}", "-c", os.path.dirname(file_path) + "/fxn.cc", "-fno-inline", "-o", fxn_file])
        compilation = subprocess.run(["g++",
                                        "-std=c++11",
                                        f"-{compilation_flag}",
                                        "-fno-inline", file_path, fxn_file,
                                        "-o", bin_file],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         text=True,
                                         timeout=300
                                     )
        if compilation.returncode != 0 or fxn_compile != 0:
            print(f"compilation on {file_path} at {compilation_flag} flag failed: {compilation.stdout}")
            continue
        else:
            print(f"compilation worked for {file_path} at {compilation_flag}")
            function_dir = join(program_directory, compilation_flag, "functions")
            extract = subprocess.run(["stoke", "extract",
                            "-i", bin_file,
                            "-o", function_dir],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=300
                           )

            if extract.returncode != 0:
                print(f"extract on {file_path} at {compilation_flag} flag failed: {extract.stdout}")
                continue
            else:
                print(f"stoke extract worked for {file_path} at {compilation_flag}")
                if compilation_flag == "O0":
                    base_asbly_path = join(function_dir, assembly_file_name + ".s")
                    tc_path = join(testcases_directory, assembly_file_name + ".tc")
                    tc = subprocess.run(['stoke', 'testcase',
                                    '--target', base_asbly_path,
                                    "-o", tc_path,
                                    '--functions', function_dir,
                                    "--prune",
                                    '--max_testcases', "1024",
                                    "--live_dangerously"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   timeout=300)
                    if tc.returncode != 0:
                        successful_testcase_flag = False
                        print(f"testcases on {file_path} at {compilation_flag} flag failed: {tc.stdout}")
                        continue
                    else:
                        successful_testcase_flag = True
                        print(f"stoke testcase generation worked for {file_path} at {compilation_flag}")

                if successful_testcase_flag:
                    test_asbly_path = join(function_dir, assembly_file_name + ".s")
                    rc, stdout, cost, correct  = test_costfn(target_f = base_asbly_path,
                                                             rewrite_f = test_asbly_path,
                                                             testcases_f = tc_path,
                                                             fun_dir = function_dir,
                                                             live_dangerously = True)
                    cost_dict[f"{compilation_flag}-O0_cost"] = float(cost)
                    cost_dict[f"{compilation_flag}-O0_correct"] = True if correct == "yes" else False

    return cost_dict



if __name__ == "__main__":
    #breakpoint()
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    with open(args.path_to_c_prog_list) as f:
        c_prog_list = f.readlines()
        c_prog_list = [c_prog.strip() for c_prog in c_prog_list]
    with open(args.path_to_asbly_prefix_list) as f:
        asbly_name_list = f.readlines()
        asbly_name_list = [asbly_name.strip() for asbly_name in asbly_name_list] 
    pre_process(c_prog_list, asbly_name_list, args.destination_dir)
