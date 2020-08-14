import json
from pymongo import MongoClient
from tqdm import tqdm
import os
import json
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import re

COLLAPSE_PATTERN = re.compile("/\./")


@dataclass
class ParseOptions:
    binary_info_out_file: str = field(metadata=dict(args=["-bin_file_out", "--binary_info_out_file"]))
    database_name: str = field(metadata=dict(args=["-db_name", "--database_name"]))
    collection_name: str = field(metadata=dict(args=["-field", "--field_name"]), default="repos")
    config_file: str = field(metadata=dict(args=["-config", "--config_file"]), default="./database-config")


def collect(binary_info_out_file: str, database_name: str, collection_name: str = "repos",
            config_file: str = "./database-config") -> None:

    assert not os.path.exists(binary_info_out_file), "the out-file specified already exists, please change or delete"

    binary_info_dict = collect_binaries(database_name=database_name, collection_name=collection_name,
                                       config_file=config_file)

    with open(binary_info_out_file, "w") as fh:
        json.dump(binary_info_dict, fh)

    return None


def collect_binaries(database_name: str, collection_name: str, config_file: str ) -> None:
    """
    collects the file names the specified database (i.e. the assembly file, the corresponding ELF file, as well as the corresponding hashes for both)

    database: database to query for results

    returns a dictionary in which the key is the full path to the assembly file and its value is a dictionary with the repo_path, original assembly_file name, corresponding ELF file name, and their respective hashes
    """
    with open(config_file, "r") as fh: 
        config = json.load(fh)

    client = MongoClient(config['host'], port=config['port'], authSource=config['auth_db_name'],
                         username=config['username'], password=config['password'])

    db_compile_results = client[database_name][collection_name]

    file_names_dictionary = {}
    results = db_compile_results.find()
    total = results.count()

    for compile_result in tqdm(results, total=total):
        if compile_result["num_binaries"] > 0:
            for makefile in compile_result["makefiles"]:
                # ensure the makefile didn't fail or return no binaries
                if makefile["success"] == True and makefile["binaries"] != [] and makefile["sha256"] != []:
                    directory = makefile["directory"]
                    orig_files = makefile["binaries"]
                    sha256_files = makefile['sha256']
                    repo_path = "/".join([compile_result["repo_owner"], compile_result["repo_name"]])

                    file2sha = {file: sha for file, sha in zip(orig_files, sha256_files)}

                    for file_name in orig_files:
                        file_hash = file2sha[file_name]

                        identifier = "/".join([repo_path, directory, file_name])
                        file_names_dictionary[identifier] = {"repo_path": repo_path,
                                                             "directory": directory,
                                                             "file_name": file_name,
                                                             "file_hash": file_hash,
                                                             }

    return file_names_dictionary

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    collect(**vars(args))
