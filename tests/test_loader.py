import unittest
from bean.datasets import TextClassificationLoader

class TestLoaders(unittest.TestCase):

    def test_swahili_news(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="swahili_news")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(len(train_data), 22207)
        self.assertEqual(dev_data, None)
        self.assertEqual(len(test_data), 7338)

    def test_hausa_voa_topics(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="hausa_voa_topics")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(len(train_data), 2045)
        self.assertEqual(len(dev_data), 290)
        self.assertEqual(len(test_data), 582)

    def test_yoruba_bbc_topics(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="yoruba_bbc_topics")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(len(train_data), 1340)
        self.assertEqual(len(dev_data), 189)
        self.assertEqual(len(test_data), 379)

    def test_swahili_tweet_sentiment(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="swahili-tweet-sentiment")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(train_data, 2263)
        self.assertEqual(dev_data, None)
        self.assertEqual(test_data, None)

    def test_yosm(self):
        classification_loader_hgf = TextClassificationLoader(dataset_name="yosm")
        train_data, dev_data, test_data = classification_loader_hgf.load_data()

        self.assertEqual(classification_loader_hgf._check_dataset_availability(), True)
        self.assertEqual(train_data, 800)
        self.assertEqual(dev_data, 200)
        self.assertEqual(test_data, 500)

        
