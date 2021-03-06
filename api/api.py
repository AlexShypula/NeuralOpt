from flask import Flask, request #import main Flask class and request object
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from stoke import StokePipeline


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
    verification_strategy: str = field(metadata=dict(args=["--strategy"]), default="hold_out") # "bounded" is the other option
    alias_strategy: str = field(metadata=dict(args=["--alias_strategy"]), default=None) # "basic" "flat"
    bound: int = field(metadata=dict(args=["--bound"]), default=None) # STOKE default is 2
    cost_timeout: int = field(metadata=dict(args=["--cost_timeout"]), default=100) #
    verification_timeout: int = field(metadata=dict(args=["--verification_timeout"]), default=300)
    # flask params
    port: int = field(metadata=dict(args=["-port", "--port"]), default=5000)
    debug: bool = field(metadata=dict(args=["-d", "--debug_mode"]), default=False)
    hack_validator: bool = field(metadata=dict(args=["--hack_validator"]), default=False)
    override_heap_out: bool = field(metadata=dict(args=["--override_heap_out"]), default=False)
    override_live_out: bool = field(metadata=dict(args=["--override_live_out"]), default=False)

@app.route('/stoke', methods = ["GET"])
def Process():
    req = request.get_json()
    jobs = req.values()
    res = pipeline.run_parallel_pipeline(jobs=jobs, debug=app.config["DEBUG"])
    res_dict = {}
    for k, v in zip(req.keys(), res):
        assert req[k]["metadata"]["hash"] == v["metadata"]["hash"]
        res_dict[k] = v
    return res_dict


@app.route('/stoke/eval', methods = ["GET"])
def Eval():
    req = request.get_json()
    jobs = req.values()
    res = pipeline.run_parallel_eval(jobs=jobs, debug=app.config["DEBUG"])
    res_dict = {}
    for k, v in zip(req.keys(), res):
        res_dict[k] = v
    return res_dict


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
    global pipeline
    pipeline = StokePipeline(n_workers = args.n_workers,
                                     max_cost = args.max_cost,
                                     verification_strategy = args.verification_strategy,
                                     path_to_volume = args.path_to_volume,
                                     volume_path_to_data = args.volume_path_to_data,
                                     volume_path_to_tmp = args.volume_path_to_tmp,
                                     alias_strategy=args.alias_strategy,
                                     bound=args.bound,
                                     cost_timeout=args.cost_timeout,
                                     verification_timeout=args.verification_timeout,
                                     hack_validator=args.hack_validator,
                                     override_heap_out=args.override_heap_out,
                                     override_live_out=args.override_live_out)
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)































