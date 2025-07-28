from datasets import *

# datasets 基本使用
# 加载在线数据集

datasets = load_dataset('/home/nanji/workspace/new-title-chinese')
print(datasets)
print('0' * 100)

# 加载数据集合集中的某一项任务
boolq_dataset = load_dataset('/home/nanji/workspace/super_glue', 'boolq', trust_remote_code=True)
print('1' * 100)
# 按照数据集划分进行加载
dataset = load_dataset('/home/nanji/workspace/new-title-chinese', split='train')
print('2' * 100)

dataset = load_dataset("/home/nanji/workspace/new-title-chinese", split="train[10:100]")
print('3' * 100)
dataset = load_dataset("/home/nanji/workspace/new-title-chinese", split="train[:50%]")
print('4' * 100)
dataset = load_dataset("/home/nanji/workspace/new-title-chinese", split=["train[:50%]", "train[50%:]"])

# 查看数据集
datasets = load_dataset("/home/nanji/workspace/new-title-chinese")

# print('5' * 100)
# print(datasets['train'][0])
# print(datasets['train'][:2])
# print(datasets['train']['title'][:5])
datasets['train'].column_names
# print('6' * 100)
datasets['train'].features

# 数据集划分
dataset = datasets['train']
dataset.train_test_split(test_size=0.1)
dataset = boolq_dataset['train']
dataset.train_test_split(test_size=0.1,
                         stratify_by_column='label')  # 分类数据集可以按照比例划分
# 数据选取与过滤
# 选取
datasets['train'].select([0, 1])
# 过滤
filter_dataset = datasets['train'].filter(lambda example: '中国' in example['title'])

a = filter_dataset['title'][:5]


# print(a)


# 数据映射
def add_prefix(example):
    example['title'] = 'Prefix:' + example['title']
    return example


prefix_dataset = datasets.map(add_prefix)
# print('6' * 100)
# print(prefix_dataset['train'][:10]['title'])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/bert-base-chinese')


def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example['content'], max_length=512, truncation=True)
    labels = tokenizer(example['title'], max_length=32, truncation=True)
    # label 就是 title 编码的结果
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# process_datasets = datasets.map(preprocess_function)
# print(process_datasets)

# process_datasets = datasets.map(preprocess_function, num_proc=4)
# print(process_datasets)

process_datasets = datasets.map(preprocess_function, batched=True)
# print('5' * 100)
# print(process_datasets)
# process_datasets = datasets.map(preprocess_function,
#                                 batched=True,
#                                 remove_columns=datasets['train'].column_names)

# print(process_datasets)

from datasets import load_from_disk

# 保存与加载
# process_datasets.save_to_disk('/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/processed_data')
process_datasets = load_from_disk(
    '/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/processed_data')
# print('6' * 100)
# print(process_datasets)
# 加载本地数据集
# 直接加载文件作为数据集
# dataset = load_dataset('csv', data_files='./ChnSentiCorp_htl_all.csv', split='train')
# dataset = Dataset.from_csv(
#     '/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/ChnSentiCorp_htl_all.csv')
# print('7' * 100)
# print(dataset)
# 加载文件夹内全部文件作为数据集
# dataset = load_dataset('csv',
#                        data_files=[
#                            '/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/all_data/ChnSentiCorp_htl_all.csv',
#                            '/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/all_data/ChnSentiCorp_htl_all copy.csv']
#                        , split='train')
# print('8' * 100)
# print(dataset)
# 加载本地数据集
# 直接加载文件作为数据集
# dataset = load_dataset('csv',
#                        data_files=[
#                            '/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/ChnSentiCorp_htl_all.csv']
#                        , split='train')
#
# print('9' * 100)
# print(dataset)
## 加载文件夹内全部文件作为数据集
# dataset = load_dataset("csv",
#                        data_files=["./all_data/ChnSentiCorp_htl_all.csv", "./all_data/ChnSentiCorp_htl_all copy.csv"],
#                        split='train')
# print(dataset)

## 通过预先加载的其他格式转换加载数据集
import pandas as pd

data = pd.read_csv('/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/ChnSentiCorp_htl_all.csv')
print('0' * 100)
print(data.head())
dataset = Dataset.from_pandas(data)
print('1' * 100)
print(dataset)
# List 格式的数据需要内嵌{},明确数据字段
data = [{'text': 'abc'}, {'text': 'def'}]
# data = ['abc', 'def']
print('2' * 100)
print(Dataset.from_list(data))
# 通过自定义加载脚本加载数据集

load_dataset('json',
             data_files='/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/cmrc2018_trial.json',
             field='data')
dataset = load_dataset('/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/', split='train')

print('3' * 100)
print(dataset[0])
# Dataset with DataCollator
from transformers import DataCollatorWithPadding

dataset = load_dataset('csv',
                       data_files='/home/nanji/workspace/transformers-code/01-Getting Started/05-datasets/ChnSentiCorp_htl_all.csv',
                       split='train')
dataset = dataset.filter(lambda x: x['review'] is not None)
print('4' * 100)
print(dataset)


def process_function(example):
    tokenizer_examples = tokenizer(example['review'], max_length=128, truncation=True)
    tokenizer_examples['label'] = example['label']
    return tokenizer_examples


tokenized_dataset = dataset.map(process_function, batched=True, remove_columns=dataset.column_names)
print('5' * 100)
print(tokenized_dataset)
print('6' * 100)
print(tokenized_dataset[:3])
collator = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader

dl = DataLoader(tokenized_dataset,
                batch_size=4,
                collate_fn=collator,
                shuffle=True)
num = 0
for batch in dl:
    print(batch['input_ids'].size())
    num += 1
    if num > 10:
        break
