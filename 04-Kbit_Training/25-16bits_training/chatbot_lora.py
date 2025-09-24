# -*- coding: utf-8 -*-
# @Time : 2025/9/24 20:54
# @Author : nanji
# @Site : 
# @File : chatbot_lora.py
# @Software: PyCharm
# @Comment :

# Lora 实战
## Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

## Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
## Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained(
    "/home/nanji/workspace/Llama-2-7b-ms",
    trust_remote_code=True)
print(tokenizer)
# tokenizer.padding_side='right'

def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join(
        ["Human: " + example["instruction"], example["input"]]).strip() + "\n\n Assistant: "  # query
    instruction = tokenizer(instruction,add_special_tokens=False)
    response = tokenizer(example["output"] + tokenizer.eos_token,add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


process_func(ds[0])

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

print(tokenized_ds)

print("0" * 100)
a = tokenizer.decode(tokenized_ds[1]['input_ids'])
print(a)

tokenizer("呀", add_special_tokens=False)
b = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
print(b)

import torch

model = AutoModelForCausalLM.from_pretrained(
    '/home/nanji/workspace/Llama-2-7b-ms', \
    low_cpu_mem_usage=True, \
    torch_dtype=torch.half)

print(model)
print("5" * 100)
for name, parameter in model.named_parameters():
    print(name)

# Lora

# PEFT Step1 配置文件
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["self_attn.q_proj"],
    modules_to_save=['word_embeddings']
)
print(config)
# PEFT Step2 配置文件
model = get_peft_model(model, config)

print(config)
# for name, parameter in model.named_parameters():
#     print(name, parameter)
print(model)

model.print_trainable_parameters()
for name, parameter in model.named_parameters():
    print(name)
# step5 配置训练参数

args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    # learning_rate=1e-4,
    # remove_unused_columns=False,
    # save_strategy="epoch"
)
## Step6 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
print("4" * 100)
