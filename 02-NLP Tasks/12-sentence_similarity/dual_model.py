# -*- coding: utf-8 -*-
# @Time : 2025/8/31 22:49
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV13P411C7UD?t=611.3
# @File : dual_model.py
# @Software: PyCharm
# @Comment :
# 文本相似度实例
# Step1 导入相关包

from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    Trainer, \
    TrainingArguments
from datasets import load_dataset

# Step2 加载数据集
dataset = load_dataset('json', data_files='./train_pair_1w.json', split='train')
print(dataset)

print(dataset[0])
# Step3 划分数据集
datasets = dataset.train_test_split(test_size=0.2)
# datasets
## Step4 数据集预处理


import torch

tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/chinese-macbert-base")


def process_function(examples):
    sentences = []
    labels = []
    for sen1, sen2, label in zip(examples['sentence1'], examples['sentence2'], examples['label']):
        sentences.append(sen1)
        sentences.append(sen2)
        labels.append(1 if int(label) == 1 else -1)
    # input_ids, attention_mask, token_type_ids
    tokenized_examples = tokenizer(sentences, max_length=128, truncation=True, padding=True)
    tokenized_examples = {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    tokenized_examples['labelss'] = labels
    return tokenized_examples


tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
print("0" * 100)
print(tokenized_datasets['train'][0])
# Step5 创建模型
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CosineSimilarity, CosineEmbeddingLoss


class DualModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        self.bert = BertModel(config)
        self.post_init()

