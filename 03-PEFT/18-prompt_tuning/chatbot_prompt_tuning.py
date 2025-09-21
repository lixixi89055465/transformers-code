# -*- coding: utf-8 -*-
# @Time : 2025/9/20 16:44
# @Author : nanji
# @Site : 
# @File : chatbot_prompt_tuning.py
# @Software: PyCharm
# @Comment :
# Prompt Tuning 实战
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
print(ds)

print(ds[:3])

# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")


# tokenizer
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    a = "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: "
    instruction = tokenizer(a)
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
print(tokenized_ds)
a = tokenizer.decode(tokenized_ds[1]['input_ids'])

b = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
# Step4 创建模
model = AutoModelForCausalLM.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")
# Prompt tuning
# PEFT Step1 配置文件
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit

# soft prompt
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
print("0" * 100)
print(config)

model = get_peft_model(model, config)
print(model)

c = model.print_trainable_parameters()
# Step5 配置训练参数
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=20
)
# Step6 创建训练
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
# Step7 模型训练
# trainer.train()
# 加载训练好的PEFT模型
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")
peft_model = PeftModel.from_pretrained(model=model, model_id="./chatbot/checkpoint-500/")
## step8 模型推理
# peft_model = peft_model.cuda()
# d = "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "
# ipt = tokenizer(d, return_tensors="pt").to(peft_model.device)
# e = peft_model.generate(**ipt, max_length=128, do_sample=True)
# f = e[0]
# g = tokenizer.decode(f, skip_special_tokens=True)
# print(g)
peft_model = peft_model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(
    peft_model.device)
print(
    tokenizer.decode(
        peft_model.generate(**ipt, max_length=128, do_sample=True)[0],
        skip_special_tokens=True
    )
)