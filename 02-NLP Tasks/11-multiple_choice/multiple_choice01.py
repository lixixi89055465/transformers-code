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
from transformers import (AutoTokenizer,
                          AutoModelForMultipleChoice, TrainingArguments, Trainer)

# Step2 加载数据集
c3 = DatasetDict.load_from_disk("./c3/")
# c3
# c3["train"][:10]
c3.pop("test")
# Step3 数据集预处理
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer = AutoTokenizer.from_pretrained('D:/workspace/chinese-macbert-base')
print(tokenizer)

def process_function(examples):
    # examples, dict, keys: ["context", "quesiton", "choice", "answer"]
    # examples, 1000
    context = []
    question_choice = []
    labels = []
    for idx in range(len(examples["context"])):
        ctx = "\n".join(examples["context"][idx])
        question = examples["question"][idx]
        choices = examples["choice"][idx]
        for choice in choices:
            context.append(ctx)
            question_choice.append(question + " " + choice)
        if len(choices) < 4:
            for _ in range(4 - len(choices)):
                context.append(ctx)
                question_choice.append(question + " " + "不知道")
        labels.append(choices.index(examples["answer"][idx]))
    tokenized_examples = tokenizer(context,
                                   question_choice,
                                   truncation="only_first",
                                   max_length=256,
                                   padding="max_length")  # input_ids: 4000 * 256,
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in
                          tokenized_examples.items()}  # 1000 * 4 *256
    tokenized_examples["labels"] = labels
    return tokenized_examples


# res = process_function(c3['train'][:10])
# print(res)

# res = c3['train'].select(range(10)).map(process_function, batched=True)
# print(res)
import numpy as np

# np.array(res["input_ids"]).shape

tokenized_c3 = c3.map(process_function, batched=True)
print(tokenized_c3)
# Step4 创建模型
# model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")
model = AutoModelForMultipleChoice.from_pretrained('D:/workspace/chinese-macbert-base')# Step5 c创建评估函数
import numpy as np

accuracy = evaluate.load("accuracy")
def compute_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(
        predictions=predictions,
        references=labels
    )


# step6 配置训练参数
args = TrainingArguments(
    output_dir='./multiple_choice',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True
)
# step7 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_c3['train'],
    eval_dataset=tokenized_c3['validation'],
    compute_metric=compute_metric
)
# step8 模型训练
trainer.train()
# step8 模型预测
from typing import Any
import torch


class MultipleChoicePipeline:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def preprocess(self, context, question, choices):
        cs, qcs = [], []
        for choice in choices:
            cs.append(context)
            qcs.append(question + '\t' + choice)
        return tokenizer(cs,
                         qcs,
                         truncation="only_first",
                         max_length=256,
                         return_tensors="pt")

    def predict(self, inputs):
        inputs = {k: v.uniqueeze(0).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def postprocess(self, logits, choices):
        prediction = torch.argmax(logits, dim=-1).cpu().item()
        return choices[prediction]

    def __call__(self, context, question, choices) -> Any:
        inputs = self.preprocess(context, question, choices)
        logits = self.predict(inputs)
        result = self.postprocess(logits, choices)
        return result


pipe = MultipleChoicePipeline(model, tokenizer)
res = pipe("小明在北京上班", "小明在哪里上班？", ["北京", "上海", "河北", "海南", "河北", "海南"])
print(res)
