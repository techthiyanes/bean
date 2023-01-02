import unittest
from bean.datasets import TextClassificationLoader


class TestLoaders(unittest.TestCase):

    def test_classification_loader(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="swahili_news")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(len(train_data), 22207)
        self.assertEqual(dev_data, None)
        self.assertEqual(len(test_data), 7338)

    def test_swahili_news(self):
        pass


