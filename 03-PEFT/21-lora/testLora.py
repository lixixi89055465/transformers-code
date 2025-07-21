# -*- coding: utf-8 -*-
# @Time : 2025/7/20 20:38
# @Author : nanji
# @Site :
# @File : testLora.py
# @Software: PyCharm
# @Comment :
'''
/home/nanji/workspace/transformers-code

'''
import pprint
import json
from datasets import Dataset
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, DataCollatorForSeq2Seq, \
    TrainingArguments, Trainer

ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
# pprint.pprint(ds[:3],indent=4)

# print(json.dumps(ds[:3]))
a = json.dumps(ds[:3])
b = json.loads(a)
print(b)

tokenizer = AutoTokenizer.from_pretrained("/home/sdb2/workspace/bloom-1b4-zh")
print('0' * 100)
print(tokenizer)


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        '\n'.join(['Human: ' + example['instruction'], example['input']]).strip() + '\n\nAssistant: ')
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
print(a)
b = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
print(b)

model = AutoModelForCausalLM.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh')
for name, parameter in model.named_parameters():
    print(name)

from peft import LoraConfig, TaskType, get_peft_model

# config = LoraConfig(task_type=TaskType.CAUSAL_LM)
# config = LoraConfig(task_type=TaskType.CAUSAL_LM,
#                     target_modules=['query_key_value', 'dense_4h_to_h'],
#                     modules_to_save=['word_embeddings'])
config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    target_modules=["*.*.1.*.*"])
# config = LoraConfig(task_type=TaskType.CAUSAL_LM,
#                     target_modules=["query_key_value"])
print(config)

model = get_peft_model(model, config)
print(config)
for name, parameter in model.named_parameters():
    print(name)

print('1' * 100)
print(model.print_trainable_parameters())

args = TrainingArguments(
    output_dir='./chatbot',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
a = trainer.train()
print(a)

model = model.cuda()
ipt = (tokenizer('Human: {}\n {}'.format("考试有哪些技巧?", "")
                 .strip()
                 + "\n\nAssistant: ", return_tensors='pt')
       .to(model.device))

a = tokenizer.decode(
    model.generate(**ipt, max_length=128, do_sample=True)[0],
    skip_special_tokens=True)

print(a)
