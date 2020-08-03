import requests
from typing import Dict, Tuple

class StokeRequest:
    def __init__(self, base_url: str = "http://127.0.0.1", port: str = "6000"):
        self.base_url = base_url
        self.port = port
        self.url = base_url + ":" + port + "/"
    # jobs should be a python dict of of format {"{job_id}": (hypothesis_string, metadata_dictionary)}
    # where job-id is a unique identifier for the job, the hypothesis string is the formatted neural net output
    # and the metadata dictionary is the metadata dictaionary that contains all necessary info for stoke processing
    def get(self, jobs: Dict[str, Tuple[str, Dict]]):
        r = requests.get(url = self.url, json = jobs)
        return r.json()
