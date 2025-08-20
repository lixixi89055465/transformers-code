# 基于 transformers 的 命名实体识别
# step1 导入相关包
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, DataCollatorForTokenClassification

# ner_datasets=load_dataset("peoples_daily_ner",cache_dir='./data')
# ner_datasets = load_dataset("/home/nanji/workspace/people-daily-ner", cache_dir='/home/nanji/workspace/people-daily-ner')
ner_datasets = load_dataset("./ner_data", cache_dir='./ner_data')

print(ner_datasets)
print('0' * 100)
print(ner_datasets['train'][0])
print('1' * 100)
print(ner_datasets['train'].features)
print('2' * 100)
label_list = ner_datasets['train'].features['ner_tags'].feature.names
print('3' * 100)
print(label_list)
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/chinese-macbert-base')
a = tokenizer(ner_datasets['train'][0]['tokens'], is_split_into_words=True)


def process_function(examples):
    tokenizer_examples = tokenizer(
        examples['text'],
        max_length=128,
        truncation=True
    )
    tokenized_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_examples.word_ids(batch_index=i)
        label_ids=[]
        for word_id in word_ids:
            pass



    return tokenizer_examples


print(ner_datasets.map(process_function, batched=False))
