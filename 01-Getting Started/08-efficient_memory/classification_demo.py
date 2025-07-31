'''
pip install transformers==3.1.0


'''
# 文本分类实例
# step1 导入相关包
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments
from datasets import load_dataset

# step2 加载数据集
dataset = load_dataset(
    'csv',
    data_files='/home/nanji/workspace/transformers-code/01-Getting Started/06-evaluate/ChnSentiCorp_htl_all.csv',
    split='train'
)

# step3 划分数据集
datasets = dataset.train_test_split(test_size=0.1)
print('1' * 100)
print(datasets)
# step4 数据集预处理
import torch

tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/rbt3')


def process_function(examples):
    if examples is None or "review" not in examples or "label" not in examples:
        print(examples)
    if examples['review'] is None or len(examples['review']) == 0:
        print(examples)
    try:
        tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    except:
        for i in examples['review']:
            if i:
                print(i)
            else:
                print(i)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples


tokenized_datasets = datasets.map(process_function, batched=False, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)
# step5 创建模型
model = AutoModelForSequenceClassification.from_pretrained('/home/nanji/workspace/rbt3')
print('3' * 100)
print(model.config)
