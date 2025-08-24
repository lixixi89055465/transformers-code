# -*- coding: utf-8 -*-
# @Time : 2025/8/22 21:32
# @Author : nanji
# @Site : 
# @File : mrc_simple_version.py
# @Software: PyCharm 
# @Comment :
# 基于截断策略的机器阅读理解任务实现
## Step1 导入相关包
# Step2 数据集加载
from datasets import load_dataset, DatasetDict
import json
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# Step2 数据集加载
# 如果可以联网，直接使用load_dataset进行加载
# datasets = load_dataset("cmrc2018", cache_dir="data")
# 如果无法联网，则使用下面的方式加载数据集
datasets = DatasetDict.load_from_disk("mrc_data")
print(datasets)
print(datasets['train'][0])

# Step3 数据预处理
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')

print(tokenizer)

sample_dataset = datasets['train'].select(range(10))
tokenized_examples = tokenizer(
    text=sample_dataset['question'],
    text_pair=sample_dataset['context'],
    return_offsets_mapping=True,
    max_length=384,
    truncation='only_second',
    padding='max_length'
)
print(tokenized_examples.data.keys())

print(tokenized_examples['input_ids'][0])
print('0' * 100)
print(len(tokenized_examples['input_ids'][0]))
print('1' * 100)
print(list(zip(tokenized_examples['input_ids'][0], tokenized_examples['token_type_ids'][0])))
print('2' * 100)
print(tokenized_examples['offset_mapping'][0])
print(len(tokenized_examples['offset_mapping'][0]))
offset_mapping = tokenized_examples.pop('offset_mapping')
for idx, offset in enumerate(offset_mapping):
    answer = sample_dataset[idx]['answers']
    start_char = answer['answer_start'][0]
    end_char = start_char + len(answer['text'][0])
    print(answer, start_char, end_char)
    # 定位答案在 token中的起始位置和结束位置
    # 一种策略，拿到 context的其实和结束，然后从左右两侧向答案逼近
    context_start = tokenized_examples.sequence_ids(idx).index(1)
    context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
    print(context_start, context_end)
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
    print(answer, start_char, end_char, context_start, context_end)


def process_func(examples):
    tokenized_examples = tokenizer(
        text=examples['question'],
        text_pair=examples['context'],
        return_offsets_mapping=True,
        max_length=384,
        truncation='only_second',
        padding='max_length'
    )
    offset_mapping = tokenized_examples.pop('offset_mapping')
    start_positions = []
    end_positions = []
    for idx, offset in enumerate(offset_mapping):
        answer = examples['answers'][idx]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        # print(answer, start_char, end_char)
        # 定位答案在 token中的起始位置和结束位置
        # 一种策略，拿到 context的其实和结束，然后从左右两侧向答案逼近
        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
        # print(context_start, context_end)
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
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)

    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    return tokenized_examples


tokenied_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)

print(tokenied_datasets)
# Step4 加载模型
model = AutoModelForQuestionAnswering.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
# step5 配置 TrainingArguments
args = TrainingArguments(
    output_dir="models_for_qa",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    num_train_epochs=3
)
# step6 配置 Trainer
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenied_datasets["train"],
    eval_dataset=tokenied_datasets["validation"],
    data_collator=DefaultDataCollator()
)
trainer.train()

# step8 模型预测
from transformers import pipeline

pipe = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)
print('1' * 100)
print(pipe)
res = pipe(question='小明在哪里上班?', context='小明在北京上班。')
print(res)
