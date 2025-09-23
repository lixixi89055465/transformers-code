# -*- coding: utf-8 -*-
# @Time : 2025/9/23 20:45
# @Author : nanji
# @Site : 
# @File : chatbot_ia3.py
# @Software: PyCharm
# @Comment :
# IA3 实战
from datasets import Dataset
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, \
    TrainingArguments, Trainer

## Step2 加载数据集
ds = Dataset.load_from_disk('../data/alpaca_data_zh/')
print(ds)
# Step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/bloom-1b4-zh")
print(tokenizer)


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer('\n'.join(['Human: ' + example['instruction'],
                                       example['input']]).strip() + '\n\n Assistant: ')
    response = tokenizer(example['output'] + tokenizer.eos_token)
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
a = tokenizer.decode(tokenized_ds[1]['input_ids'])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['input_ids'])))

## Step4 创建模型
model = AutoModelForCausalLM.from_pretrained('/home/nanji/workspace/bloom-1b4-zh')
## IA3
### PEFT Step1 配置文件
from peft import IA3Config, TaskType, get_peft_model

config = IA3Config(task_type=TaskType.CAUSAL_LM)
print(config)

### PEFT Step2 创建模型
model = get_peft_model(model, config)
print(model)
model.print_trainable_parameters()
print("1" * 100)
# Step5 配置训练参数
args = TrainingArguments(
    output_dir='./chatbot',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=3e-3
)
# Step6 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True
    )
)
## Step7 模型训练

trainer.train()

# Step8 模型推理
model = model.cuda()
ipt = tokenizer(
    "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ",
    return_tensors="pt").to(model.device)
b = tokenizer.decode(
    model.generate(**ipt, max_length=128, do_sample=True)[0],
    skip_special_tokens=True
)
