import logging
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class AutoModel:
    """
    A class to load a model and tokenizer for training and evaluation
    """
    def __init__(self, model_name_or_path: str, tokenizer_name: str):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name = tokenizer_name

    def classification_model(self, num_labels: str, label2id: dict, seed: int, use_slow_tokenizer: bool=False ):
        """
        Load a classification model
        input: num_labels (int) - the number of labels in the dataset
                label2id (dict) - the mapping of labels to ids
                seed (int) - the seed to use for reproducibility
                use_slow_tokenizer (bool) - whether to use the slow tokenizer
        output: model (transformers.modeling_utils.PreTrainedModel) - the classification model
                tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase) - the tokenizer
        """
        self._set_seed(seed)
        config = AutoConfig.from_pretrained(self.model_name_or_path, num_labels=num_labels,
                                            label2id=label2id, id2label={v: k for k, v in label2id.items()})
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name or self.model_name_or_path, use_fast=not use_slow_tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
        )
        return model, tokenizer

    def ner_model(self):
        pass

    def _set_seed(self, seed):
        """
        Set the seed for reproducibility
        input: seed (int) - the seed to use for reproducibility
        """
        set_seed(seed)
