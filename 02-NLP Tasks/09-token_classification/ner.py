# 基于 transformers 的 命名实体识别
# step1 导入相关包
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, DataCollatorForTokenClassification

# ner_datasets=load_dataset("peoples_daily_ner",cache_dir='./data')
ner_datasets = load_dataset("/home/nanji/workspace/people-daily-ner",
                            cache_dir='/home/nanji/workspace/people-daily-ner')
# ner_datasets=load_dataset("./ner_data_new",cache_dir='./ner_data_new')

print(ner_datasets)
print('0' * 100)
print(ner_datasets['train'][0])
print('1' * 100)
print(ner_datasets['train'].features)
print('2' * 100)
print(ner_datasets['train'].features['entities'])
print('3' * 100)
print(print(ner_datasets['train'][0]['text']))
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
print('4' * 100)
print(tokenizer(ner_datasets['train'][0]['text']))
res = tokenizer('interesting word is good')

print(res)
print('5' * 100)
print(res.word_ids())


def process_function(examples):
    tokenizer_examples = tokenizer(
        examples['text'],
        max_length=128,
        truncation=True
    )
    for item in examples['entities']:
        pass
    return tokenizer_examples


print(ner_datasets.map(process_function, batched=False))
