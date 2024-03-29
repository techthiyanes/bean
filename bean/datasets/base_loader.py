import yaml
import requests
from os import path

DATASET_CONFIG = path.join(path.dirname(__file__), 'config/all_datasets.yaml')

class BaseLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_data(self):
        pass

    def _check_dataset_availability(self) -> bool:
        """
        Check whether the specified dataset exists on huggingface
        output: 
            bool - True if the dataset exists, False otherwise
        """
        try:
            url = f"https://huggingface.co/datasets/{self.dataset_name}"
            response = requests.get(url)
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as ex:
            print(ex)
            self.error_message = "Can not send a request to https://huggingface.co"
            return False

    def _check_available_splits(self) -> list:
        """
        List of all the available splits for a particular dataset
        returns: 
            list - the list of available splits
        """
        url = f"https://datasets-server.huggingface.co/splits?dataset={self.dataset_name}"
        response = requests.request("GET", url)
        return [data_split['split'] for data_split in response.json()['splits']]

    def get_dataset_metadata(self, task: str) -> list:
        """
        Get the metadata of a particular dataset
        input: 
            task (str) - the task the dataset is used for
        returns: 
            list - the metadata of the dataset
        """
        data_dict = yaml.safe_load(open(DATASET_CONFIG))
        return [dataset for dataset in data_dict[task] if dataset["name"]==self.dataset_name][0]

