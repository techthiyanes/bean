import torch
import logging
import evaluate
from bean.evaluate import BaseEvaluate
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator)


# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

text_classification_metrics = ['accuracy', 'f1', 'precision', 'recall']

class TextClassificationEvaluate(BaseEvaluate):
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, **kwargs ):
        super(BaseEvaluate).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.__dict__.update(kwargs)

        self.accelerator = self._initialize_accelerator()

    def evaluate(self):
        pass

    def inference(self, data_loader: DataLoader):
        self.model.eval()
        samples_seen = 0

        for step, batch in enumerate(data_loader):
            with torch.no_grad():
                outputs = self.model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = self.accelerator.gather((predictions, batch["labels"]))
        if self.accelerator.num_processes > 1:
            if step == len(data_loader) - 1:
                predictions = predictions[: len(data_loader.dataset) - samples_seen]
                references = references[: len(data_loader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        return predictions, references

    def load_metrics(self):
        metrics = [evaluate.load(metric) for metric in text_classification_metrics]
        return metrics

    def _create_dataloader(self, dataset_collection):
        # DataLoaders creation:
        if self.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))
        dataloader = DataLoader(
            dataset_collection, shuffle=True, collate_fn=data_collator, batch_size=self.batch_size
        )
        return dataloader

    def process_dataset_split(self, dataset_collection, split):
        with self.accelerator.main_process_first():
            processed_datasets = dataset_collection.map(
                self._tokenize_dataset,
                batched=True,
                remove_columns=dataset_collection[split].column_names,
                desc="Running tokenizer on dataset",
            )

        return processed_datasets

    def _tokenize_dataset(self, examples):
        # Tokenize the texts
        texts = ((examples[self.text_column_name],))
        result = self.tokenizer(*texts, padding=self.padding, max_length=self.max_length, truncation=self.truncation)

        if "label" in examples:
            if self.label_to_id is not None:
                result["labels"] = [self.label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    def _initialize_accelerator(self):
        accelerator = (
            Accelerator(log_with=self.report_to, logging_dir=self.output_dir) if self.with_tracking else Accelerator()
        )

        return accelerator
    
