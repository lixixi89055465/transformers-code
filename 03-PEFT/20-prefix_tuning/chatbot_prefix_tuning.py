# -*- coding: utf-8 -*-
# @Time : 2025/9/21 13:20
# @Author : nanji
# @Site : 
# @File : chatbot_prefix_tuning.py
# @Software: PyCharm
# @Comment :
# Prefix-Tuning 实战
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")


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

a = tokenizer.decode(tokenized_ds[1]['input_ids'])
b = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
model = AutoModelForCausalLM.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")
## Prefix-tuning
### PEFT Step1 配置文件
from peft import PrefixTuningConfig, get_peft_model, TaskType

config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                            num_virtual_tokens=10,
                            prefix_projection=True)

print(config)
# PEFT Step2 创建模型

model = get_peft_model(model, config)
print(model.prompt_encoder)
model.print_trainable_parameters()
## Step5 配置训练参数
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)
## Step6 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, padding=True
    )
)
## Step7 模型训练
trainer.train()
## Step8 模型推理
model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() +
                "\n\nAssistant: ", return_tensors="pt").to(model.device)
g = tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
print(g)
