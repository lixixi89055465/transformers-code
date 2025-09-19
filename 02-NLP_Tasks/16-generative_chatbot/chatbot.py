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
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM,DataCollatorForSeq2Seq,TrainingArguments,Trainer
# Step2 加载数据集
ds = Dataset.load_from_disk("./alpaca_data_zh/")
# ds

print(ds[:3])
# Step3 数据集预处理
tokenizer=AutoTokenizer.from_pretrained("/home/nanji/workspace/bloom-389m-zh")
print('0'*100)
print(tokenizer)

