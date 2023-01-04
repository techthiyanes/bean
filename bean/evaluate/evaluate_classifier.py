import torch
import logging
import evaluate
from tqdm import tqdm
from datasets import Dataset
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
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
        self.metrics = self._load_metrics()

    def evaluate(self, dataset: Dataset):
        """

        """
        tokenized_dataset = self.process_dataset_split(dataset)
        dataloader = self._create_dataloader(tokenized_dataset)
        predictions, references = self.inference(dataloader)

        precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="macro")
        acc = accuracy_score(references, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


    def inference(self, data_loader: DataLoader):
        """
        Generate prediction and ground-truth target pair from eval dataloader
        """
        self.model.eval()
        samples_seen = 0
        pred_list, reference_list = [], []
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
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
            pred_list.extend(predictions.tolist())
            reference_list.extend(references.tolist())

        assert len(pred_list) == len(reference_list)

        return pred_list, reference_list

    def _load_metrics(self) -> List:
        """
        load metrics for evaluation
        """
        metrics = [evaluate.load(metric, average='weighted') for metric in text_classification_metrics]
        return metrics

    def _create_dataloader(self, tokenized_dataset: Dataset) -> DataLoader:
        """
        Create a dataloader from the tokenized dataset
        tokenized_dataset: Dataset ==> tokenized dataset
        returns:
        dataloader: DataLoader ==> DataLoader
        """
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
            tokenized_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.batch_size
        )
        return dataloader

    def process_dataset_split(self, dataset_collection: Dataset) -> Dataset:
        """
        This function uses the _tokenize_dataset method to tokenize and map dataset labels
        dataset_collection: Dataset ==> raw dataset
        returns:
        processed_datasets: Dataset ==> tokenized Dataset
        """
        with self.accelerator.main_process_first():
            processed_datasets = dataset_collection.map(
                self._tokenize_dataset,
                batched=True,
                remove_columns=dataset_collection.column_names,
                desc="Running tokenizer on dataset",
            )

        return processed_datasets

    def _tokenize_dataset(self, examples: Dataset) -> Dataset:
        """
        This function tokenizes the raw datasets and maps the target feature to ids
        examples: Datasets ==> raw text dataset
        returns:
        result: Datasets ==>  tokenized dataset
        """
        # Tokenize the texts
        texts = ((examples[self.text_feature_name],))
        result = self.tokenizer(*texts, padding=self.padding, max_length=self.max_length, truncation=self.truncation)

        if "label" in examples:
            if not self.label_mapped:
                result["labels"] = [self.label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[self.label_feature_name]
        return result

    def _initialize_accelerator(self) -> Accelerator:
        """
        Initialize accelerator; used for managing device placement for easy distribution
        """
        accelerator = Accelerator()
        return accelerator
    
