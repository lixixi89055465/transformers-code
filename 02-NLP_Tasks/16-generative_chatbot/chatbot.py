# -*- coding: utf-8 -*-
# @Time : 2025/9/19 17:49
# @Author : nanji
# @Site : 
# @File : chatbot.py
# @Software: PyCharm 
# @Comment :
# 生成式对话机器人
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Step2 加载数据集
ds = Dataset.load_from_disk("./alpaca_data_zh/")
# ds

print(ds[:3])
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/bloom-389m-zh")
print('0' * 100)
print(tokenizer)


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
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


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenized_ds
tokenizer.decode(tokenized_ds[1]["input_ids"])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))
# Step4 创建模型
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/home/nanji/workspace/bloom-389m-zh",
    trust_remote_code=True)
## Step5 配置训练参数
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=2
)
## Step6 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
## Step7 模型训练
# trainer.train()
# Step8 模型推理
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "
a = pipe(ipt, max_length=256, do_sample=True, )
print(a)
