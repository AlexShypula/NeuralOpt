import requests
from typing import Dict, Tuple

class StokeRequest:
    def __init__(self, base_url: str = "http://127.0.0.1", port: str = "6000"):
        if type(port) == int:
            port = str(port)

        self.base_url = base_url
        self.port = port
        self.url = base_url + ":" + port + "/"
        self.stoke_url = self.url + "stoke"
        self.eval_url = self.url + "stoke/eval"
        print("a new requester was initialized, the dest is {}".format(self.stoke_url))
    # jobs should be a python dict of of format {"{job_id}": (hypothesis_string, metadata_dictionary)}
    # where job-id is a unique identifier for the job, the hypothesis string is the formatted neural net output
    # and the metadata dictionary is the metadata dictaionary that contains all necessary info for stoke processing
    def get(self, jobs: Dict[str, Tuple[str, Dict]]):
        r = requests.get(url = self.stoke_url, json = jobs)
        return r.json()

    def eval(self, jobs: Dict[str, Tuple[str, Dict]]):
        r = requests.get(url=self.eval_url, json=jobs)
        return r.json()
