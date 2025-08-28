# -*- coding: utf-8 -*-
# @Time : 2025/8/28 22:11
# @Author : nanji
# @Site : 
# @File : multiple_choice.py
# @Software: PyCharm 
# @Comment :
# 基于Transformers的多项选择
# Step1 导入相关包
import evaluate
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer

# Step2 加载数据集
c3 = DatasetDict.load_from_disk('./c3/')
print(c3)
print(c3['train'][:10])
print(c3.pop('test'))
print(c3)
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
print('0' * 100)
print(tokenizer)


def process_function(examples):
    # examples, dict, keys: ["context", "quesiton", "choice", "answer"]
    # examples, 1000
    context = []
    question_choice = []
    labels = []
    for idx in range(len(examples['context'])):
        ctx = '\n'.join(examples['context'][idx])
        question = examples['question'][idx]
        choices = examples['choice'][idx]
        for choice in choices:
            context.append(ctx)
            question_choice.append(question + ' ' + choice)
        if len(choices) < 4:
            for _ in range(4 - len(choices)):
                context.append(ctx)
                question_choice.append(question + " " + "不知道")
        labels.append(choices.index(examples['answer'][idx]))
    tokenized_examples = tokenizer(context,
                                   question_choice,
                                   truncation='only_first',
                                   max_length=256,
                                   padding='max_length')  # input_ids : 4000 * 256
    tokenized_examples = {
        k: [v[i:i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }
    tokenized_examples['labels'] = labels
    return tokenized_examples


res = c3['train'].select(range(10)).map(process_function, batched=True)
print(res)
import numpy as np

print(np.array(res['input_ids']).shape)

