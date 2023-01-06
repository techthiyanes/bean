import os
import math
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
    default_data_collator,
    get_scheduler)


# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

text_classification_metrics = ['accuracy', 'f1', 'precision', 'recall']

class TextClassificationEvaluate(BaseEvaluate):
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, **kwargs ):
        super(BaseEvaluate).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.__dict__.update(kwargs)

        self.accelerator = self._initialize_accelerator()
        self.metrics = self._load_metrics()

    
    def finetune_evaluate(self, train_dataset: Dataset, eval_dataset: Dataset, config: dict):
        """
        Finetune the model on the train dataset and evaluate on the eval dataset
        input: train_dataset (Dataset) - the train dataset
                eval_dataset (Dataset) - the eval dataset
                config (dict) - the configuration for training
        returns: metrics (dict) - the evaluation metrics
        """
        self._finetune(train_dataset, config)
        return self.evaluate(eval_dataset)

    def evaluate(self, eval_dataset: Dataset):
        """
        Evaluate the model on the eval dataset
        input: eval_dataset (Dataset) - the eval dataset
        returns: metrics (dict) - the evaluation metrics
        """
        tokenized_dataset = self.process_dataset_split(eval_dataset)
        dataloader = self._create_dataloader(tokenized_dataset)
        predictions, references = self.inference(dataloader)

        precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="macro")
        acc = accuracy_score(references, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


    def inference(self, data_loader: DataLoader):
        """
        Run inference on the model
        input: data_loader (DataLoader) - the dataloader for the dataset
        returns: predictions (List) - the predictions
                references (List) - the references
        """
        self.model, data_loader = self.accelerator.prepare(self.model, data_loader)
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

    def _finetune(self, train_dataset: Dataset, config: dict):
        """
        Finetune the model on the train dataset
        input: train_dataset (Dataset) - the train dataset
                config (dict) - the configuration for training
        """
        tokenized_dataset = self.process_dataset_split(train_dataset)
        train_dataloader = self._create_dataloader(tokenized_dataset)
        optimizer = self._create_optimizer(config)
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.get("gradient_accumulation_steps", 1))
        if config.get("max_train_steps", None) is None:
            config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = self._get_linear_schedule_with_warmup(optimizer, config["max_train_steps"], config)

        self.model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(self.model, optimizer, train_dataloader, lr_scheduler)
        # We need to recalculate our total training steps as the size of the training dataloader may have changed
        
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
        if overrode_max_train_steps:
            config["max_train_steps"]= config["num_train_epochs"] * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        config["num_train_epochs"] = math.ceil(config["max_train_steps"]/ num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps =  config.get("checkpointing_steps", None)
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        
        # Train!
        total_batch_size = config["train_batch_size"] * self.accelerator.num_processes * config["gradient_accumulation_steps"]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {config['num_train_epochs']}")
        logger.info(f"  Instantaneous batch size per device = {config['train_batch_size']}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
        logger.info(f"  Total optimization steps = {config['max_train_steps']}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(config["max_train_steps"]), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if config.get("resume_from_checkpoint", None):
            if config.get("resume_from_checkpoint", None) is not None or config.get("resume_from_checkpoint", None) != "":
                self.accelerator.print(f"Resumed from checkpoint: {config['resume_from_checkpoint']}")
                self.accelerator.load_state(config['resume_from_checkpoint'])
                path = os.path.basename(config['resume_from_checkpoint'])
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        for epoch in range(starting_epoch, config["num_train_epochs"]):
            self.model.train()
            if config["with_tracking"]:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if config.get("resume_from_checkpoint", None) and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = self.model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if config["with_tracking"]:
                    total_loss += loss.detach().float()
                loss = loss / config["gradient_accumulation_steps"]
                self.accelerator.backward(loss)
                if step % config["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if config["output_dir"] is not None:
                            output_dir = os.path.join(config["output_dir"], output_dir)
                        self.accelerator.save_state(output_dir)

                if completed_steps >= config['max_train_steps']:
                    break
            
            logger.info(f"Loss after epoch: {epoch} is {total_loss/len(train_dataloader)}")


    def _load_metrics(self) -> List:
        """
        Load the metrics
        returns:
        metrics: List ==> List of metrics
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
        This function tokenizes the dataset
        examples: Dataset ==> raw dataset
        returns:
        result: Dataset ==> tokenized dataset
        """
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
        Initialize accelerator
        returns:
        accelerator: Accelerator ==> accelerator
        """
        accelerator = Accelerator(logging_dir=self.output_dir) if self.output_dir else Accelerator()
        return accelerator
    
    def _create_optimizer(self, config: dict):
        """
        Create optimizer
        config: dict ==> configuration dictionary
        returns:
        optimizer: torch.optim.Optimizer ==> optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

        return optimizer

        
    def _get_linear_schedule_with_warmup(self, optimizer: torch.optim.Optimizer, num_training_steps: int, config: dict):
        """
        Create a scheduler with a learning rate that decreases after increasing during a warmup period.
        inputs:
        optimizer: torch.optim.Optimizer ==> optimizer
        num_training_steps: int ==> number of training steps
        config: dict ==> config parameters
        returns:
        lr_scheduler: torch.optim.lr_scheduler ==> scheduler
        """
        lr_scheduler = get_scheduler(
            name=config["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=config["num_warmup_steps"],
            num_training_steps=num_training_steps,
        )

        return lr_scheduler
    
    def _validate_config(self, config: dict):
        """
        Validate the configuration parameters
        config: dict ==> configuration parameters
        returns:
        config: dict ==> validated configuration parameters
        """
        return config
    
