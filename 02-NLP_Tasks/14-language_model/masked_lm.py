# -*- coding: utf-8 -*-
# @Time : 2025/9/14 14:34
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1B44y1c7x2?t=2.9
# @File : masked_lm.py
# @Software: PyCharm
# @Comment :
# 掩码语言模型训练实例

# Step1 导入相关包
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, \
    DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer

# Step2 加载数据
ds = Dataset.load_from_disk('./wiki_cn_filtered/')
print(ds)
print("0" * 100)
print(ds[0])
# Step3 数据集处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/chinese-macbert-base")


def process_func(examples):
    return tokenizer(examples['completion'], max_length=384)


tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)

print(tokenized_ds)
from torch.utils.data import DataLoader

dl = DataLoader(tokenized_ds,
                batch_size=2,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer, mlm=True, mlm_probability=0.15
                ))

print(next(enumerate(dl)))

print(tokenizer.mask_token)
print("0" * 100)
print(tokenizer.mask_token_id)
print("1" * 100)

## Step4 创建模型
model = AutoModelForMaskedLM.from_pretrained("/home/nanji/workspace/chinese-macbert-base")
# Step5 配置训练参数
args = TrainingArguments(
    output_dir='./masked_lm',
    per_device_train_batch_size=32,
    logging_steps=10,
    num_train_epochs=1
)
# Step6 创建训练器
trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
)
# Step7 模型训练
trainer.train()
## Step8 模型推理
from transformers import pipeline

pipe = pipeline(
    'fill-mask',
    model=model,
    tokenizer=tokenizer,
    device=0
)
p1 = pipe("西安交通[MASK][MASK]博物馆（Xi'an Jiaotong University Museum）是一座位于西安交通大学的博物馆")
print(p1)
p2 = pipe("下面是一则[MASK][MASK]新闻。小编报道，近日，游戏产业发展的非常好！")
print(p2)
