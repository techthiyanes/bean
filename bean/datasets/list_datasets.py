import json
from typing import List, Any

DATASETS_LIST = "bean/datasets/config/all_datasets.json"

def list_datasets(task: str = None, language: str=None) -> Any:
    with open(DATASETS_LIST) as f:
        data_dict = json.load(f)

    if task:
        assert task in data_dict.keys(), f"The specified task ({task}) does not exist"
        return data_dict[task]

    if language:
