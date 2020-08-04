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
    verification_strategy: str = field(metadata=dict(args=["strategy"]), default="hold_out")
    # flask params
    port: int = field(metadata=dict(args=["-port", "--port"]), default=5000)
    debug: bool = field(metadata=dict(args=["-d", "--debug_mode"]), default=False)


@app.route('/stoke', methods = ["GET"])
def Process():
    req = request.get_json()
    jobs = req.values()
    res = pipeline.run_parallel_pipeline(jobs=jobs, debug=app.config["DEBUG"])
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
                                     volume_path_to_tmp = args.volume_path_to_tmp)
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)































