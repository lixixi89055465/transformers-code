# 基于 transformers 的 命名实体识别
'''

pip install seqeval

'''
# step1 导入相关包
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, DataCollatorForTokenClassification
# 设置CUDA环境变量以减少内存碎片化（可选）
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# ner_datasets=load_dataset("peoples_daily_ner",cache_dir='./data')
# ner_datasets = load_dataset("/home/nanji/workspace/people-daily-ner", cache_dir='/home/nanji/workspace/people-daily-ner')
ner_datasets = load_dataset("./ner_data", cache_dir='./ner_data')

print(ner_datasets)
print('0' * 100)
print(ner_datasets['train'][0])
print('1' * 100)
print(ner_datasets['train'].features)
print('2' * 100)
label_list = ner_datasets['train'].features['ner_tags'].feature.names
print('3' * 100)
print(label_list)
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
a = tokenizer(ner_datasets['train'][0]['tokens'], is_split_into_words=True)
b = tokenizer('interesting word')
print(b)
c = b.word_ids()
print(c)


def process_function(examples):
    tokenized_exmaples = tokenizer(
        examples['tokens'],
        max_length=128,
        truncation=True,
        is_split_into_words=True
    )
    # tokenized_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_exmaples["labels"] = labels
    return tokenized_exmaples


tokenized_datasets = ner_datasets.map(process_function, batched=True)
tokenized_datasets

# 对于所有的非二分类任务，切记要指定num_labels，否则就会device错误
model = AutoModelForTokenClassification.from_pretrained(
    "/home/nanji/workspace/chinese-macbert-base",
    num_labels=len(label_list))
model.config.num_labels

## Step5 创建评估函数
# 这里方便大家加载，替换成了本地的加载方式，无需额外下载
seqeval = evaluate.load("seqeval_metric.py")
seqeval

import numpy as np


def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

    return {
        "f1": result["overall_f1"]
    }


## Step6 配置训练参数

args = TrainingArguments(
    output_dir="models_for_ner",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=50,
    num_train_epochs=1
)
## Step7 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)
## Step8 模型训练
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])
## Step9 模型预测
from transformers import pipeline

# 使用pipeline进行推理，要指定id2label
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
model.config
# 如果模型是基于GPU训练的，那么推理时要指定device
# 对于NER任务，可以指定aggregation_strategy为simple，得到具体的实体的结果，而不是token的结果
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")
res = ner_pipe("小明在北京上班")
res
# 根据start和end取实际的结果
ner_result = {}
x = "小明在北京上班"
for r in res:
    if r["entity_group"] not in ner_result:
        ner_result[r["entity_group"]] = []
    ner_result[r["entity_group"]].append(x[r["start"]: r["end"]])

ner_result
