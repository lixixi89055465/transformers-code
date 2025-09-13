# -*- coding: utf-8 -*-
# @Time : 2025/9/13 20:31
# @Author : nanji
# @Site : 
# @File : retrieval_bot.py
# @Software: PyCharm
# @Comment :
# 检索机器人
import pandas as pd

data = pd.read_csv("./law_faq.csv")
print(data.head())
## Step2 加载模型
from dual_model import DualModel

# 需要完成前置模型训练
dual_model = DualModel.from_pretrained("../12-sentence_similarity/dual_model/checkpoint-500/")
dual_model = dual_model.cuda()
dual_model.eval()
print("匹配模型加载成功！")
from transformers import AutoTokenizer

tokenzier = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
print(tokenzier)
# Step3 将问题编码为向量
import torch
from tqdm import tqdm

questions = data["title"].to_list()
vectors = []
with torch.inference_mode():
    for i in tqdm(range(0, len(questions), 32)):
        batch_sens = questions[i: i + 32]
        inputs = tokenzier(batch_sens, return_tensors="pt", padding=True, max_length=128, truncation=True)
        inputs = {k: v.to(dual_model.device) for k, v in inputs.items()}
        vector = dual_model.bert(**inputs)[1]
        vectors.append(vector)
vectors = torch.concat(vectors, dim=0).cpu().numpy()
print(vectors.shape)
## Step4 创建索引
import faiss

index = faiss.IndexFlatIP(768)
faiss.normalize_L2(vectors)
index.add(vectors)
print(index)
## Step5 对问题进行向量编码

quesiton = "寻衅滋事"
with torch.inference_mode():
    inputs = tokenzier(quesiton, return_tensors="pt", padding=True, max_length=128, truncation=True)
    inputs = {k: v.to(dual_model.device) for k, v in inputs.items()}
    vector = dual_model.bert(**inputs)[1]
    q_vector = vector.cpu().numpy()
print(q_vector.shape)
## Step6 向量匹配(召回)

faiss.normalize_L2(q_vector)
scores, indexes = index.search(q_vector, 10)
topk_result = data.values[indexes[0].tolist()]
print(topk_result[:, 0])

## Step7 加载交互模型
