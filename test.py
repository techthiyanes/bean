from bean.datasets import TextClassificationLoader
from bean.models import AutoModel


swahili_news_dataset = TextClassificationLoader(dataset_name="swahili_news")
train_dataset, dev_dataset, test_dataset = swahili_news_dataset.load_data()

dataset_config = swahili_news_dataset.get_dataset_metadata(task="text_classification")


print(dataset_config)
model_class = AutoModel(model_name_or_path="castorini/afriberta_small", tokenizer_name="castorini/afriberta_small" )
model = model_class.classification_model(num_labels=dataset_config['num_labels'], label2id=dataset_config['label2id'])

print(model)