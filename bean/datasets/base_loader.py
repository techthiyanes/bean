import requests

class BaseLoader:
    def __init__(self, dataset_name: str):
        pass

    def _check_dataset_availability(self) -> bool:
        """
        Check whether the specified dataset exists
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

    def _check_available_splits(self, split: str):
        """
        """
        url = f"https://datasets-server.huggingface.co/splits?dataset={self.dataset_name}"
        response = requests.request("GET", url)
        return [data_split['split'] for data_split in response.json()['splits']]