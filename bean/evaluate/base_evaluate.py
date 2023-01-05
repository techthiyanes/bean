from abc import abstractmethod

class BaseEvaluate:
    def __init__(self):
        pass

    @abstractmethod
    def preprocess_dataset(self):
        pass

    @abstractmethod
    def load_metrics(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def inference(self):
        pass