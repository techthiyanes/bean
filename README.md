# BeAN 🫘
BEnchmarking Africa NLP, A framework for easy evaluation of language models on several Africa NLP datasets


### List all datasets

You can list all the available datasets by specifying a particular task or language; see below for example

```python3
from bean.datasets.list_datasets import list_datasets

print(list_datasets(task='text_classification'))

## Returns
[{'name': 'swahili_news', 'languages_covered': ['swahili'], 'available_on_huggingface': True, 'dataset_link': 'https://huggingface.co/datasets/swahili_news'}, {'name': 'swahili_news', 'languages_covered': ['swahili'], 'available_on_huggingface': True, 'dataset_link': 'https://huggingface.co/datasets/swahili_news'}]
```

## 🫘 Onboarded Datasets

### Classification

| Dataset     | Website | Paper    | Public | Split | Download Link| languages |
| :---        |    :----:   |          ---: |  ---: |  ---: |  ---: | ---: |
| swahili_news      | [website](https://doi.org/10.5281/zenodo.5514203)       | -   | Yes | `train` `dev` | [link](https://huggingface.co/datasets/swahili_news) | `sw` |