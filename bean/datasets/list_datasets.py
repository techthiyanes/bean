import yaml
from typing import List, Any

DATASET_CONFIG = "bean/datasets/config/all_datasets.yaml"

def list_datasets(task: str = None, language: str=None) -> Any:
    data_dict = yaml.safe_load(open(DATASET_CONFIG))

    if task:
        assert task in data_dict.keys(), f"The specified task ({task}) does not exist"
        return data_dict[task]

    if language:
        assert task, f"You need to specify a task for language {language}"
        return [dataset for dataset in data_dict[task] if language in dataset["languages_covered"]]