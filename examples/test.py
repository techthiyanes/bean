from bean.datasets import TextClassificationLoader
from bean.models import AutoModel
from bean.evaluate import TextClassificationEvaluate


swahili_news = TextClassificationLoader(dataset_name="swahili_news")
train_dataset, dev_dataset, test_dataset = swahili_news.load_data()

print(dev_dataset)
print(train_dataset)
print(test_dataset)

"""
Dataset({
    features: ['news_title', 'label', 'date', 'bbc_url_id'],
    num_rows: 189
})

"""
dataset_config = swahili_news.get_dataset_metadata(task="text_classification")


print(dataset_config)


model_class = AutoModel(model_name_or_path="Davlan/afro-xlmr-small", tokenizer_name="Davlan/afro-xlmr-small")
model, tokenizer = model_class.classification_model(num_labels=dataset_config['num_labels'], label2id=dataset_config['label2id'], seed=0)

evaluation_configs={
    'label_to_id': dataset_config['label2id'],
    'max_length': 128,
    'truncation': True,
    'text_feature_name': dataset_config['text_feature_name'],
    'label_feature_name': dataset_config['label_feature_name'],
    'batch_size': 64,
    'padding': 'max_length',
    'pad_to_max_length': True,
    'label_mapped': dataset_config['label_mapped'],
    'output_dir': "models/afroxlmr-base"
}
evaluator = TextClassificationEvaluate(model=model,tokenizer=tokenizer, **evaluation_configs)


finetune_configs = {
    'num_train_epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 0,
    'weight_decay': 0.0,
    'gradient_accumulation_steps': 1,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1.0,
    'logging_steps': 100,
    'checkpointing_steps': '1000',
    'train_batch_size': 64,
    'lr_scheduler_type': 'linear',
    'num_warmup_steps': 0,
    'output_dir': "models/afroxlmr-base",
    'seed': 0,
    'with_tracking': True
}

scores = evaluator.finetune_evaluate(train_dataset=train_dataset, eval_dataset=test_dataset, config=finetune_configs)
print(scores)