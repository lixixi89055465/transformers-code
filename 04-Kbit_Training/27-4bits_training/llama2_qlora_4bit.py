# -*- coding: utf-8 -*-
# @Time : 2025/9/29 20:52
# @Author : nanji
# @Site :  https://www.bilibili.com/video/BV1DQ4y1t7e8?t=1161.8
# @File : llama2_qlora_4bit.py
# @Software: PyCharm
# @Comment :
# Lora 实战
# Step1 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
print("0" * 100)
print(ds[:3])
# print(len("以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"))

# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/Llama-2-7b-ms")
# tokenizer
tokenizer.padding_side = "right"  # 一定要设置padding_side为right，否则batch大于1时可能不收敛
tokenizer.pad_token_id = 2


def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer( \
        "\n".join(
            ["Human: " + example["instruction"],
             example["input"]]
        ).strip() + "\n\nAssistant: ",
        add_special_tokens=False
    )
    response = tokenizer(example["output"],
                         add_special_tokens=False)
    input_ids = instruction["input_ids"] + \
                response["input_ids"] + \
                [tokenizer.eos_token_id]
    attention_mask = (instruction["attention_mask"] + \
                      response["attention_mask"] + [1])
    labels = [-100] * len(instruction["input_ids"]) \
             + response["input_ids"] \
             + [tokenizer.eos_token_id]
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
# tokenized_ds
print(tokenized_ds[0]["input_ids"])
# tokenizer("abc " + tokenizer.eos_token)
tokenizer.decode(tokenized_ds[0]["input_ids"])
# tokenizer("呀", add_special_tokens=False) # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))
# Step4 创建模型
import torch

# 多卡情况，可以去掉device_map="auto"，否则会将模型拆开
model = AutoModelForCausalLM.from_pretrained( \
    "/home/nanji/workspace/Llama-2-7b-ms", \
    low_cpu_mem_usage=True, \
    torch_dtype=torch.bfloat16, \
    device_map="auto", \
    load_in_4bit=True, \
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", \
    bnb_4bit_use_double_quant=True)
for name, param in model.named_parameters():
    print(name, param.shape, param.dtype)
## Lora
### PEFT Step1 配置文件
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(task_type=TaskType.CAUSAL_LM, )
print(config)

print("2" * 100)
print(model.config)

# PEFT Step2 创建模型
model = get_peft_model(model, config)
print(config)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
# torch.tensor(1e-8).half()
model.print_trainable_parameters()
## Step5 配置训练参数
args = TrainingArguments(
    output_dir='./chatbot',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    logging_steps=10,
    num_train_epochs=1,
    gradient_checkpointing=True,
    optim='paged_adamw_32bit'
)
## Step6 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
## Step7 模型训练
trainer.train()
## Step8 模型推理
model.eval()
ipt = tokenizer("Human: {}\n{}".format("你好", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0],
                 skip_special_tokens=True)
print("5" * 100)
model.merge_and_unload()
