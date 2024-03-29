import yaml
from typing import List, Any

DATASET_CONFIG = "datasets/config/all_datasets.yaml"

def list_datasets(task: str = None, language: str=None) -> Any:
    """
    List all the available datasets
    input: task (str) - the task the dataset is used for,
              language (str) - the dataset language
    returns: 
        list - the list of available datasets
    """
    data_dict = yaml.safe_load(open(DATASET_CONFIG))

    if task:
        assert task in data_dict.keys(), f"The specified task ({task}) does not exist"
        return data_dict[task]

    if language:
        assert task, f"You need to specify a task for language {language}"
        return [dataset for dataset in data_dict[task] if language in dataset["languages_covered"]]