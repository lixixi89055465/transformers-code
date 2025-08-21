# 基于 transformers 的 命名实体识别
'''

pip install seqeval

'''
# step1 导入相关包
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
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
b = tokenizer('interesting word')
print(b)
c = b.word_ids()
print(c)


def process_function(examples):
    tokenized_exmaples = tokenizer(
        examples['tokens'],
        max_length=128,
        truncation=True,
        is_split_into_words=True
    )
    # tokenized_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    input_ids=examples['input_ids']
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(input_ids[word_id])
        labels.append(label_ids)
    tokenized_exmaples["labels"] = labels
    return tokenized_exmaples


# d = ner_datasets.map(process_function, batched=False)
# print(d)
print('4' * 100)

# 对于所有的非二分类任务，切记要指定num_labels，否则就会device错误
model = AutoModelForTokenClassification.from_pretrained(
    "/home/nanji/workspace/chinese-macbert-base",
    num_labels=len(label_list))
