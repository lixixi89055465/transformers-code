# -*- coding: utf-8 -*-
# @Time : 2025/9/13 10:36
# @Author : nanji
# @Site : 
# @File : retrieval_bot.py
# @Software: PyCharm 
# @Comment :
# 检索机器人
import pandas as pd

data = pd.read_csv('../13-retrieval_chatbot/law_faq.csv')

print(data.head())
# step2 加载模型
from dual_model import DualModel

# 需要完成前置模型训练
dual_model = DualModel.from_pretrained("./12-sentence_similarity/dual_model/checkpoint-500/")
dual_model.eval()
print("匹配模型加载成功!")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
print('0' * 100)

print(tokenizer)
## Step3 将问题编码为向量
import torch
from tqdm import tqdm

questions = data['title'].to_list()
vectors = []
with torch.inference_mode():
    for i in tqdm(range(0, len(questions), 32)):
        batch_sens = questions[i:i + 32]
        inputs = tokenizer(batch_sens, return_tensors='pt', \
                           padding=True, truncation=True, \
                           max_length=128)
        inputs = {
            k: v.to(dual_model.device)
            for k, v in inputs.items()
        }
        vector=dual_model.bert(**inputs)
        vectors=vector.detach().cpu().numpy()

vectors=torch.concat(vectors,dim=0).cpu().numpy()
print(vectors.shape)
