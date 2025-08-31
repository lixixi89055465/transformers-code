# -*- coding: utf-8 -*-
# @Time : 2025/8/31 9:32
# @Author : nanji
# @Site : 
# @File : cross_model.py
# @Software: PyCharm
# @Comment :
# 文本相似度实例
# step1 导入相关包
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:48"
from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    TrainingArguments, \
    Trainer
from datasets import load_dataset

# Step2 加载数据集
dataset = load_dataset("json", data_files="./train_pair_1w.json", split="train")
print(dataset)
print(dataset[0])
# Step3 划分数据集
datasets = dataset.train_test_split(test_size=0.2)
print("0" * 100)
print(datasets)
# Step4 数据集预处理
import torch

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/home/nanji/workspace/chinese-macbert-base")


def process_function(examples):
    # tokenizer(examples['sen'])
    tokenized_examples = tokenizer(examples['sentence1'],
                                   examples['sentence2'],
                                   max_length=128,
                                   truncation=True)
    tokenized_examples['labels'] = [float(label) for label in examples['label']]
    return tokenized_examples


tokenized_datasets = datasets.map(process_function,
                                  batched=True,
                                  remove_columns=datasets["train"].column_names)

print("1" * 100)
print(tokenized_datasets["train"][0])

## Step5 创建模型
from transformers import BertForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "/home/nanji/workspace/chinese-macbert-base",
    num_labels=1)

## Step6 创建评估函数
import evaluate

acc_metric = evaluate.load('./metric_accuracy.py')
f1_metric = evaluate.load('./metric_f1.py')


def eval_metric(eval_predict):
    print("0" * 100)
    print(eval_predict)
    predictions, labels = eval_predict
    print("1" * 100)
    print(predictions)
    print("2" * 100)
    print(labels)
    predictions = [int(p > 0.5) for p in predictions]
    labels = [int(l) for l in labels]
    # predictions.
    # predictions = predictions.argmax(axis=-1)
    print(predictions[0])
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    # if 1 == 1:
    #     exit(0)
    return acc


eval_predict = (
    [
        [0.0164619],
        [0.39939108],
        [0.6035842],
    ],
    [
        0., 0., 0
    ]
)
# eval_metric(eval_predict)

# Step7 创建TrainingArguments
train_args = TrainingArguments(output_dir="./cross_model",  # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               per_device_eval_batch_size=32,  # 验证时的batch_size
                               logging_steps=10,  # log 打印的频率
                               eval_strategy="epoch",  # 评估策略
                               save_strategy="epoch",  # 保存策略
                               save_total_limit=3,  # 最大保存数
                               learning_rate=2e-5,  # 学习率
                               weight_decay=0.01,  # weight_decay
                               metric_for_best_model="f1",  # 设定评估指标
                               load_best_model_at_end=True)  # 训练完成后加载最优模型
print(train_args)
## Step8 创建Trainer
from transformers import DataCollatorWithPadding

trainer = Trainer(model=model,
                  args=train_args,
                  tokenizer=tokenizer,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
# Step9 模型训练
trainer.train()
# Step10 模型评估
trainer.evaluate(tokenized_datasets["test"])
# Step11 模型预测
from transformers import pipeline, TextClassificationPipeline

model.config.id2label = {0: '不相似', 1: '相似'}
pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)
result = pipe({
    "text": "我喜欢北京",
    "text_pair": "天气怎样"
}, function_to_apply="none")
result["label"] = '相似' if result["score"] > 0.5 else "不相似"
print(result)
