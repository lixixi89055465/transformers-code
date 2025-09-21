# -*- coding: utf-8 -*-
# @Time : 2025/9/21 12:13
# @Author : nanji
# @Site : 
# @File : chatbot_ptuning.py
# @Software: PyCharm
# @Comment :
# P-tuning 实战
## Step1 导入相关包
from datasets import Dataset

from transformers import AutoTokenizer, \
    AutoModelForCausalLM, DataCollatorForSeq2Seq, \
    TrainingArguments, Trainer

# Step2 加载数据集
ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
# ds
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/bloom-1b4-zh")


# tokenizer
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    a = '\n'.jsoin(['Human: ' + example['instruction']]).strip() + '\n\n Assistant: '
    instruction = tokenizer(a)
    b = example['output'] + tokenizer.eos_token
    response = tokenizer(b)
    input_ids = instruction['input_ids'] + response['input_ids']
    attention_mask = instruction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instruction['input_ids']) + response['input_ids']
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_ds)
c = tokenizer.decode(tokenized_ds[1]['input_ids'])

d = tokenizer.decode(
    list(
        filter(
            lambda x: x != -100, tokenized_ds[1]['labels']
        )
    )
)
## Step4 创建模型
model = AutoModelForCausalLM.from_pretrained(
    '/home/nanji/workspace/bloom-1b4-zh'
)
# P-tuning
# PEFT Step1 配置文件
from peft import PromptEncoderConfig, \
    TaskType, \
    get_peft_model, \
    PromptEncoderReparameterizationType

config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM, \
    num_virtual_tokens=10, \
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP, \
    encoder_dropout=0.1, \
    encoder_num_layers=5, \
    encoder_hidden_size=1024
)
print(config)
### PEFT Step2 创建模型

model = get_peft_model(model, config)
print(config)
model.print_trainable_parameters()
## Step5 配置训练参数
args = TrainingArguments(
    output_dir='./chatbot',
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
)
# Step7 模型训练
trainer.train()
## Step8 模型推理

e = 'Human: {}\n{}'.format('数学技巧有那些技巧?', '').strip() + '\n\n Assistant: '
ipt = tokenizer(
    e,
    return_tensor='pt'
).to(model.device)
f = tokenizer.decode(
    model.generate(
        **ipt,
        max_length=256,
        do_sample=True
    )[0],
    skip_special_tokens=True
)
print(f)
