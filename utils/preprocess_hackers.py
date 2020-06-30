import subprocess
from typing import List
import os
from os import makedirs
from os.path import splitext, join
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


@dataclass
class ParseOptions:
    path_to_c_prog_list: str = field(metadata=dict(args=["-c_progs", "--path_to_c_prog_list"]))
    path_to_asbly_prefix_list: str = field(metadata=dict(args=["-asbly_names", "--path_to_asbly_prefix_list"]))
    destination_dir: str = field(metadata=dict(args=["-out_dir", "--out_directory"], default="processed_data"))


def mkdir(dir:str):
    if not os.path.exists(dir):
        makedirs(dir)


def pre_process(c_progs: List[str], asbly_prefixes: List[str],
                destination_dir: str = "processed_data",
                compilation_flags: List[str] = ["O0", "Og", "O2", "O3"]):
    for c_prog_pth, asbly_prefix in zip(c_progs, asbly_prefixes):
        process_program(file_path=c_prog_pth,
                        assembly_file_name = asbly_prefix,
                        destination_dir=destination_dir,
                        compilation_flags = compilation_flags)


def process_program(file_path: str,
                    assembly_file_name: str,
                    destination_dir: str = "processed_data",
                    compilation_flags: List[str] = ["O0", "Og", "O2", "O3"],
                    ):
    #breakpoint()
    file_prefix = splitext(file_path)[0]
    #program_directory = join(destination_dir, file_prefix)
    program_directory = join(destination_dir, os.path.basename(os.path.dirname(file_path)))
    testcases_directory = join(program_directory, "testcases")
    ### make directories
    mkdir(testcases_directory)
    for compilation_flag in compilation_flags:
        for subfolder in ("functions", "bin"):
            mkdir(join(program_directory, compilation_flag, subfolder))
        fxn_file =  join(program_directory, compilation_flag, "bin", "fxn.o")
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
        if compilation.returncode != 0:
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
                    asbly_path = join(function_dir, assembly_file_name + ".s")
                    tc_path = join(testcases_directory, assembly_file_name + ".tc")
                    tc = subprocess.run(['stoke', 'testcase',
                                    '--target', asbly_path,
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
                        print(f"testcases on {file_path} at {compilation_flag} flag failed: {tc.stdout}")
                        continue
                    else:
                        print(f"stoke testcase generation worked for {file_path} at {compilation_flag}")

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
