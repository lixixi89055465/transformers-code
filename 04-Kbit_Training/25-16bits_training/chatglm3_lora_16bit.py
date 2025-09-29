# -*- coding: utf-8 -*-
# @Time : 2025/9/29 15:35
# @Author : nanji
# @Site : 
# @File : chatglm3_lora_16bit.py
# @Software: PyCharm 
# @Comment :
# Lora 实战
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
print(ds)
print(ds[:3])
# print(len("以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"))
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained( \
    "/home/nanji/workspace/chatglm3-6b-base", \
    trust_remote_code=True)
print('0' * 100)


# print(tokenizer(tokenizer.eos_token), tokenizer.eos_token_id)

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join([example["instruction"], \
                             example["input"]]).strip()
    instruction = tokenizer.build_chat_input( \
        instruction, \
        history=[], \
        role='user')
    response = tokenizer('\n' + example['output'], add_special_tokens=False)
    input_ids = instruction["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0].numpy().tolist()) + response['input_ids'] + [
        tokenizer.eos_token_id]
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
tokenizer.decode(tokenized_ds[1]['input_ids'])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
# Step4 创建模型
import torch

"""
新版本中需要将modeling_chatglm源码中的613行部分进行调整，代码如下：

```
if not kv_caches:
    kv_caches = [None for _ in range(self.num_layers)]
else:
    kv_caches = kv_caches[1]
```

如果不进行调整，后续chat阶段会报错
"""
model = AutoModelForCausalLM.from_pretrained( \
    "/home/nanji/workspace/chatglm3-6b-base",
    trust_remote_code=True, \
    low_cpu_mem_usage=True, \
    torch_dtype=torch.half,
    device_map='auto',
    load_in_8bit=True
)
for name, param in model.named_parameters():
    print(name)

## Lora
# PEFT Step1 配置文件
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

config = LoraConfig(target_modules=['query_key_value'],
                    modules_to_save=['post_attention_layernorm'])
print(config)
# PEFT Step2 创建模型


model = get_peft_model(model, config)
print('1' * 100)
print(config)
for name, parameter in model.named_parameters():
    print(name)

import torch

args = TrainingArguments(
    output_dir='./chatbot',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=1,
    adam_epsilon=1e-4, remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True
    )
)
# step 7 模型训练
trainer.train()
# 模型推理
model.eval()
print(model.chat(tokenizer, '数学考试怎么考高分?', history=[])[0])
