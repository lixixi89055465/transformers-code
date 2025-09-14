# -*- coding: utf-8 -*-
# @Time : 2025/9/14 16:24
# @Author : nanji
# @Site : 
# @File : causal_lm.py
# @Software: PyCharm
# @Comment :
# 因果语言模型训练实例
## Step1 导入相关包
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, \
    TrainingArguments, \
    Trainer, \
    BloomForCausalLM

# Step2 加载数据
ds = Dataset.load_from_disk('./wiki_cn_filtered')
print(ds)
print(ds[0])
print("0" * 100)

# Step3 数据集处
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/bloom-389m-zh")


def process_func(examples):
    contents = [e + tokenizer.eos_token for e in examples['completion']]
    return tokenizer(contents, truncation=True, max_length=384)


tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)
print(tokenized_ds)
from torch.utils.data import DataLoader

dl = DataLoader(tokenized_ds, batch_size=2,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False)
                )
print(next(enumerate(dl)))

print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.eos_token, tokenizer.eos_token_id)
# ## Step4 创建模型
model = AutoModelForCausalLM.from_pretrained("/home/nanji/workspace/bloom-389m-zh")
# Step5 配置训练参数
args = TrainingArguments(
    output_dir='./causal_lm',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=1,
    fp16=True
)
## Step6 创建训练器

trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)
# Step7 模型训练
trainer.train()
## Step8 模型推理
from transformers import pipeline

pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer,
                device=0)

p1 = pipe("西安交通大学博物馆（Xi'an Jiaotong University Museum）是一座位于西安", max_length=128,
          do_sample=True)
print("0" * 100)
print(p1)
p2 = pipe("下面是一则游戏新闻。小编报道，近日，游戏产业发展的非常", max_length=128, do_sample=True)
print("1" * 100)
print(p2)
