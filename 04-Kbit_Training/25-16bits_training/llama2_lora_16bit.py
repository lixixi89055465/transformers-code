# -*- coding: utf-8 -*-
# @Time : 2025/9/25 22:08
# @Author : nanji
# @Site : 
# @File : llama2_lora_16bit.py
# @Software: PyCharm
# @Comment :
# Lora 实战
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
print(ds[:3])
print("0" * 100)
print(
    len("以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"))
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/Llama-2-7b-ms")
print(tokenizer)
tokenizer.padding_side = 'right'  # # 一定要设置padding_side为right，否则batch大于1时可能不收敛
tokenizer.pad_token_id = 2


def process_func(example):
    MAX_LENGTH = 1024  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ", \
        add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
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
