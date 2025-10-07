# -*- coding: utf-8 -*-
# @Time : 2025/10/7 22:25
# @Author : nanji
# @Site : 
# @File : classification_demo_pytorch.py
# @Software: PyCharm
# @Comment :
# 文本分类实例
# Step1 导入相关包
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, BertTokenizer, \
    BertForSequenceClassification
from datasets import load_dataset

## Step2 加载数据集
data = pd.read_csv("./ChnSentiCorp_htl_all.csv")

print(data)
data.dropna()
print(data)
# Step3 划分数据集 Datasets
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
for i in range(5):
    print(dataset[i])
## Step4 数据集预处理
from torch.utils.data import random_split

trainset, validset = random_split(dataset, lengths=[0.9, 0.1])
print(len(trainset), len(validset))
for i in range(10):
    print(trainset[i])

## Step5 创建DataLoader
import torch

tokenizer = BertTokenizer.from_pretrained('/home/nanji/workspace/chinese-roberta-wwm-ext')


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, \
                       padding='max_length', \
                       truncation=True, \
                       return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    return inputs


from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)
a = next(enumerate(validloader))
print(a)
# step6 创建模型及优化器
from torch.optim import Adam

model = AutoModelForSequenceClassification.from_pretrained('/home/nanji/workspace/rbt3')
if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)


# step 7 训练与验证
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    return acc_num / len(validset)


def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f'ep : {ep},global_step : {global_step},loss: {output.loss.item()}')
            global_step += 1
        acc = evaluate()
        print(f'ep: {ep},acc:{acc}')


train()
# step9 模型预测
sen = '我觉得这家酒店不错，饭很好吃 '
id2_label = {0: "差评", 1: "好评!"}
model.eval()
with torch.inference_mode():
    inputs=tokenizer(sen,return_tensors='pt')
