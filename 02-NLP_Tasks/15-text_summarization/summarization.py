# -*- coding: utf-8 -*-
# @Time : 2025/9/15 22:44
# @Author : nanji
# @Site : 
# @File : summarization.py
# @Software: PyCharm
# @Comment :
# 基于T5的文本摘要
## Step1 导入相关包
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments

ds = Dataset.load_from_disk("./nlpcc_2017/")
print(ds)
ds.train_test_split(100, seed=42)
print(ds)
# Step3 数据处理


tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/mengzi-t5-base")


def process_func(examples):
    contents = ["摘要生产:\n" + e for e in examples['content']]
    inputs = tokenizer(contents, max_length=384, truncation=True)
    labels = tokenizer(text_target=examples['title'], max_length=64, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs


tokenized_ds = ds.map(process_func, batched=True)
print(tokenized_ds)
print("1" * 100)
t1 = tokenizer.decode(tokenized_ds['train'][0]["input_ids"])
t2 = tokenizer.decode(tokenized_ds['train'][0]['labels'])
## Step4 创建模型
model = AutoModelForSeq2SeqLM.from_config(
    "/home/nanji/workspace/mengzi-t5-base"
)
# Step5 创建评估函数
import numpy as np
from rouge_chinese import Rouge

rouge = Rouge()


def compute_metric(evalPred):
    predictions, labels = evalPred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )
    decode_preds = [" ".join(p) for p in decode_preds]
    decode_labels = [" ".join(p) for p in decode_labels]
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
    return {
        "rouge-1": scores['rouge-1']['f'],
        "rouge-2": scores['rouge-2']['f'],
        "rouge-l": scores['rouge-l']['f'],
    }


# Step6 配置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="./summary",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    logging_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="rouge-l",
    predict_with_generate=True
)
## Step7 创建训练器
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    compute_metrics=compute_metric,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)
# Step8 模型训
trainer.train()
# Step9 模型推
from transformers import pipeline

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)
pipe('摘要生成:\n' + ds['test'][-1]['content'], max_length=64, do_sample=True)
print("0" * 100)

print(ds['test'][-1]['title'])
