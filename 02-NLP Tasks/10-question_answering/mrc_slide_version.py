# -*- coding: utf-8 -*-
# @Time : 2025/8/23 22:22
# @Author : nanji
# @Site : 
# @File : mrc_slide_version.py.py
# @Software: PyCharm 
# @Comment :
'''
import nltk
nltk.download('punkt_tab')

'''
# 基于截断策略的机器阅读理解任务实
# Step1 导入相关包
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          TrainingArguments,
                          Trainer,
                          DefaultDataCollator)
# print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
## Step2 数据集加载

# 如果可以联网，直接使用load_dataset进行加载
# datasets = load_dataset("cmrc2018", cache_dir="data")
# 如果无法联网，则使用下面的方式加载数据集
datasets = DatasetDict.load_from_disk("mrc_data")
# print(datasets)
# print('0' * 100)
# print(datasets['train'][0])
# Step3 数据预处理
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
# print('1' * 100)
# print(tokenizer)
sample_dataset = datasets['train'].select(range(10))
tokenized_examples = tokenizer(
    text=sample_dataset['question'],
    text_pair=sample_dataset['context'],
    return_offsets_mapping=True,
    return_overflowing_tokens=True,
    stride=128,
    max_length=384,
    truncation='only_second',
    padding='max_length',
)

# print(tokenized_examples.data.keys())
# print(tokenized_examples['offset_mapping'][0], len(tokenized_examples['offset_mapping'][0]))
# offset_mapping = tokenized_examples.pop('offset_mapping')

# print(tokenized_examples['overflow_to_sample_mapping'])
# print(len(tokenized_examples['overflow_to_sample_mapping']))
# for sen in tokenizer.batch_decode(tokenized_examples['input_ids'][:3]):
#     print(sen)

sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')


# for idx, _ in enumerate(sample_mapping):
#     answer = sample_dataset['answers'][sample_mapping[idx]]
#     start_char = answer['answer_start'][0]
#     end_char = start_char + len(answer['text'][0])
#     # 定位答案在 token 中的其实位置和结束位置
#     # 一种策略，我们要拿到 context 的起始和 结束，然后从左右两侧向答案逼近
#     context_start = tokenized_examples.sequence_ids(idx).index(1)
#     context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
#     offset = tokenized_examples.get('offset_mapping')[idx]
#     example_ids = []
#     # 判断答案是否在 context 中
#     if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
#         start_token_pos = 0
#         end_token_pos = 0
#     else:
#         token_id = context_start
#         while token_id <= context_end and offset[token_id][0] < start_char:
#             token_id += 1
#         start_token_pos = token_id
#         token_id = context_end
#         while token_id >= context_start and offset[token_id][1] > end_char:
#             token_id -= 1
#         end_token_pos = token_id
#         example_ids.append([sample_mapping[idx]])
#     print(answer, start_char, end_char, context_start, context_end, start_token_pos, end_token_pos)
#     print('token answer decode:',
#           tokenizer.decode(tokenized_examples['input_ids'][idx][start_token_pos:end_token_pos + 1]))


def process_func(examples):
    tokenized_examples = tokenizer(
        text=examples['question'],
        text_pair=examples['context'],
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        stride=128,
        max_length=384,
        truncation='only_second',
        padding='max_length')
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    start_positions = []
    end_positions = []
    example_ids = []
    for idx, _ in enumerate(sample_mapping):
        answer = examples['answers'][sample_mapping[idx]]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        # 定位答案在 token 中的其实位置和结束位置
        # 一种策略 ，我们要拿到 context的起始和结束，然后从左右两侧向答案逼近
        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
        offset = tokenized_examples.get('offset_mapping')[idx]
        # 判断答案是否在 context 中
        if offset[context_start][1] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
        example_ids.append(examples['id'][sample_mapping[idx]])
        tokenized_examples['offset_mapping'][idx] = [
            (o if tokenized_examples.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(tokenized_examples['offset_mapping'][idx])
        ]
    tokenized_examples['example_ids'] = example_ids
    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    return tokenized_examples


tokenized_datasets = datasets.map(
    process_func,
    batched=True,
    remove_columns=datasets['train'].column_names
)
# print('0' * 100)
# print(tokenized_datasets)
# print('1' * 100)
# print(tokenized_datasets['train']['offset_mapping'][1])
# print('2' * 100)
# print(tokenized_datasets['train']['example_ids'][:10])

import collections

# example 和 feature 的映射
example_to_feature = collections.defaultdict(list)
for idx, example_id in enumerate(tokenized_datasets['train']['example_ids'][:10]):
    example_to_feature[example_id].append(idx)
# print('3' * 100)
# print(example_to_feature)
# Step4 获取模型输出
import numpy as np
import collections

import numpy as np
import collections


def get_result(start_logits, end_logits, exmaples, features):
    predictions = {}
    references = {}

    # example 和 feature的映射
    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features["example_ids"]):
        example_to_feature[example_id].append(idx)

    # 最优答案候选
    n_best = 20
    # 最大答案长度
    max_answer_length = 30

    for example in exmaples:
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_idx in example_to_feature[example_id]:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offset = features[feature_idx]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    answers.append({
                        "text": context[offset[start_index][0]: offset[end_index][1]],
                        "score": start_logit[start_index] + end_logit[end_index]
                    })
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
        references[example_id] = example["answers"]["text"]

    return predictions, references


## Step5 评估函数
from cmrc_eval import evaluate_cmrc


def metirc(pred):
    start_logits, end_logits = pred[0]
    if start_logits.shape[0] == len(tokenized_datasets["validation"]):
        p, r = get_result(start_logits, end_logits, datasets["validation"], tokenized_datasets["validation"])
    else:
        p, r = get_result(start_logits, end_logits, datasets["test"], tokenized_datasets["test"])
    return evaluate_cmrc(p, r)


## Step6 加载模型
model = AutoModelForQuestionAnswering.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
# Step7 配置TrainingArguments
args = TrainingArguments(
    output_dir='models_for_qa',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy='steps',
    eval_steps=2,
    save_strategy='epoch',
    logging_steps=50,
    num_train_epochs=1
)
## Step8 配置Trainer
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=DefaultDataCollator(),
    compute_metrics=metirc
)
# Step9 模型训练
trainer.train()
from transformers import pipeline

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
print(pipe)

res = pipe(question="小明在哪里上班？", context="小明在北京上班")
print(res)
