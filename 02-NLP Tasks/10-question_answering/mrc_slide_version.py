# -*- coding: utf-8 -*-
# @Time : 2025/8/23 22:22
# @Author : nanji
# @Site : 
# @File : mrc_slide_version.py.py
# @Software: PyCharm 
# @Comment :
# 基于截断策略的机器阅读理解任务实
# Step1 导入相关包
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          TrainingArguments,
                          Trainer,
                          DefaultDataCollator)

## Step2 数据集加载

# 如果可以联网，直接使用load_dataset进行加载
# datasets = load_dataset("cmrc2018", cache_dir="data")
# 如果无法联网，则使用下面的方式加载数据集
datasets = DatasetDict.load_from_disk("mrc_data")
print(datasets)
print('0' * 100)
print(datasets['train'][0])
# Step3 数据预处理
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
print('1' * 100)
print(tokenizer)
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

print(tokenized_examples.data.keys())

print(tokenized_examples['offset_mapping'][0], len(tokenized_examples['offset_mapping'][0]))
# offset_mapping = tokenized_examples.pop('offset_mapping')

print(tokenized_examples['overflow_to_sample_mapping'])
print(len(tokenized_examples['overflow_to_sample_mapping']))
# for sen in tokenizer.batch_decode(tokenized_examples['input_ids'][:3]):
#     print(sen)

sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
for idx, _ in enumerate(sample_mapping):
    answer = sample_dataset['answers'][sample_mapping[idx]]
    start_char = answer['answer_start'][0]
    end_char = start_char + len(answer['text'][0])
    # 定位答案在 token 中的其实位置和结束位置
    # 一种策略，我们要拿到 context 的起始和 结束，然后从左右两侧向答案逼近
    context_start = tokenized_examples.sequence_ids(idx).index(1)
    context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
    offset = tokenized_examples.get('offset_mapping')[idx]
    example_ids = []
    # 判断答案是否在 context 中
    if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
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
        example_ids.append([sample_mapping[idx]])
    print(answer, start_char, end_char, context_start, context_end, start_token_pos, end_token_pos)
    print('token answer decode:',
          tokenizer.decode(tokenized_examples['input_ids'][idx][start_token_pos:end_token_pos + 1]))

