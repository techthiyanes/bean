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

| Dataset     | Website | Paper    | Public | Datasize | Split | Download Link| languages |
| :---        |    :----:   |          ---: |  ---: |  ---: |  ---: |  ---: | ---: |
| swahili_news                  | [website](https://doi.org/10.5281/zenodo.5514203)       | -   | Yes | `train` `dev` | [link](https://huggingface.co/datasets/swahili_news) | `sw` |
| hausa_voa_topics              | -   | -   | Yes | `train` `dev` `test` | [link](https://huggingface.co/datasets/hausa_voa_topics) | `ha` |
| yoruba_bbc_topics             | -   | -   | Yes | `train` `dev` `test` | [link](https://huggingface.co/datasets/yoruba_bbc_topics) | `yo` |
| Davis/Swahili-tweet-sentiment | [website](https://github.com/Davisy/Swahili-Tweet-Sentiment-Analysis-App)       | -   | Yes | | [link](https://huggingface.co/datasets/Davis/Swahili-tweet-sentiment) | `sw` |
| Iyanuoluwa/YOSM               | [website](https://github.com/IyanuSh/YOSM)       | [paper](https://arxiv.org/abs/2204.09711)   | Yes | `train` `dev` `test` | [link](https://huggingface.co/datasets/Iyanuoluwa/YOSM) | `yo` |




