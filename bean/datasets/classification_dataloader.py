import os
import logging
from datasets import load_dataset, Value, Features

from bean.datasets.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class TextClassificationLoader(BaseLoader):
    def __init__(self, dataset_name: str=None, train_file: str=None, test_file: str=None, dev_file: str=None):
        super().__init__(dataset_name=dataset_name)
        self.dataset_name =  dataset_name
        self.train_file =  train_file
        self.test_file =  test_file
        self.dev_file =  dev_file

    def load_data(self):
        """
        Load the dataset
        """
        train_dataset, dev_dataset, test_dataset = None, None, None

        assert self._check_dataset_availability() == True, "The specified dataset does not exist"
        if not self.dataset_name:
            assert os.path.exists(self.train_file), "Specified Train file does not exist"
            assert os.path.exists(self.test_file), "Specified Test file does not exist"
            assert os.path.exists(self.dev_file), "Specified Dev file does not exist"

            if self.train_file.endswith(".json"):
                train_dataset = load_dataset("json", data_files=[self.train_file])
                test_dataset = load_dataset("json", data_files=[self.test_file])
                dev_dataset = load_dataset("json", data_files=[self.dev_file])
            elif self.train_file.endswith(".csv"):
                train_dataset = load_dataset("csv", data_files=[self.train_file])
                test_dataset = load_dataset("csv", data_files=[self.test_file])
                dev_dataset = load_dataset("csv", data_files=[self.dev_file])
        else:
            available_splits = self._check_available_splits()
            if 'train' in available_splits:
                train_dataset = load_dataset(self.dataset_name, split='train')

            if 'test' in available_splits:
                test_dataset = load_dataset(self.dataset_name, split='test')

            if 'dev' in available_splits:
                dev_dataset = load_dataset(self.dataset_name, split='dev')
            elif 'validation' in available_splits:
                dev_dataset = load_dataset(self.dataset_name, split='validation')

        return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    loader = TextClassificationLoader(dataset_name="swahili_news")
    print(loader.load_data())
    print(loader.get_dataset_metadata(task="text_classification"))