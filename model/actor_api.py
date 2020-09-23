from flask import Flask, request #import main Flask class and request object
from flask.json import jsonify
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import multiprocessing as mp
from vocabulary import Vocabulary

from helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, bpe_postprocess, BucketReplayBuffer

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from torchtext import data
from actor_api_utils import parse_config
from copy import copy
from actor import actor_wrapper


#create the Flask app
app = Flask(__name__)


@dataclass
class ParseOptions:
    # stoke pipeline params
    volume_path_to_data: str = field(metadata=dict(args=["-path_to_data", "--volume_path_to_data"]))
    volume_path_to_tmp: str = field(metadata=dict(args=["-path_to_tmp", "--volume_path_to_tmp"]))
    path_to_volume: str = field(metadata=dict(args=["-path_to_volume", "--abs_path_to_volume"]),
                                    default='/home/stoke/docker')
    n_workers: int = field(metadata=dict(args=["-n_workers", "--number_of_workers"]), default=1)
    max_cost: float = field(metadata=dict(args=["-max_cost", "--max_cost"]), default = 9999.0)
    verification_strategy: str = field(metadata=dict(args=["-strategy"]), default="hold_out")
    # flask params
    port: int = field(metadata=dict(args=["-port", "--port"]), default=5000)
    debug: bool = field(metadata=dict(args=["-d", "--debug_mode"]), default=False)

#
# @app.route('/stoke', methods = ["GET"])
# def Process():
#     req = request.get_json()
#     jobs = req.values()
#     res = pipeline.run_parallel_pipeline(jobs=jobs, debug=app.config["DEBUG"])
#     res_dict = {}
#     for k, v in zip(req.keys(), res):
#         res_dict[k] = v
#     return res_dict


@app.route('/actor/get_experiences', methods = ["GET"])
def ClearQueue():
    #req = request.get_json()
    experiences = []
    while not traj_queue.empty():
        experiences.append(traj_queue.get())

    return jsonify(experiences)

@app.route('/actor/start', methods = ["POST"])
def generate_trajs():
    #req = request.get_json()
    if not generate_trajectory_flag.is_set():
        generate_trajectory_flag.set()
        return jsonify(True)
    else:
        return jsonify(False)

@app.route('/actor/stop', methods = ["POST"])
def stop_generating():
    #req = request.get_json()
    if generate_trajectory_flag.is_set():
        generate_trajectory_flag.clear()
        for p in processes:
            p.terminate()
        return jsonify(True)
    else:
        return jsonify(False)


@app.route('/actor/new_model_flag', methods=["POST"])
def stop_generating():
    # req = request.get_json()
    with model_id_flag.get_lock():
        model_id_flag.value += 1
    return jsonify(True)

@app.route('/actor/running_starts_over', methods=["GET"])
def stop_generating():
    # req = request.get_json()
    if actors_doing_running_starts.value == 0:
        return jsonify(True)
    else:
        return jsonify(False)

if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()

    settings_dict, actor_data_prefixes, actor_device_list = parse_config(args.config_path)

    global model_id_flag
    global traj_queue
    global actors_doing_running_starts
    global processes
    global generate_trajectory_flag
    model_id_flag = mp.Value("i", 0)
    traj_queue = mp.Queue(maxsize=1000)
    actors_doing_running_starts = mp.Value("i", settings_dict["n_actors"])
    generate_trajectory_flag = mp.Event()
    settings_dict["model_id_flag"] = model_id_flag
    settings_dict["traj_queue"] = traj_queue
    settings_dict["running_starts_counter"] = actors_doing_running_starts
    settings_dict["generate_trajs_flag"] = generate_trajectory_flag
    #generate_trajectory_flag.set()
    jobs = []
    for i, (actor_data_path_prefix, actor_device) in enumerate(zip(actor_data_prefixes, actor_device_list)):
        actor_kwargs = copy(settings_dict)
        actor_kwargs["path_to_data"] = actor_data_path_prefix
        actor_kwargs["device"] = actor_device
        actor_kwargs["actor_id"] = i

    processes = [mp.Process(target=actor_wrapper, args=(job,)) for job in jobs]
    for p in processes:
        p.start()

    app.run(debug=args.debug, host="0.0.0.0", port=args.port)































