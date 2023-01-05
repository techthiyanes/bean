from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer)

class AutoModel:
    def __init__(self, model_name_or_path: str, tokenizer_name: str):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name = tokenizer_name

    def classification_model(self, num_labels: str, label2id: dict, seed: int, use_slow_tokenizer: bool=False ):
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
        set_seed(seed)
