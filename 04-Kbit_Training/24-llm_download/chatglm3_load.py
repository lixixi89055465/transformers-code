# -*- coding: utf-8 -*-
# @Time : 2025/9/23 21:56
# @Author : nanji
# @Site : 
# @File : chatglm3_load.py
# @Software: PyCharm
# @Comment :
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/chatglm3-6b/", trust_remote_code=True)
print(tokenizer)
model = AutoModel.from_pretrained("/home/nanji/workspace/chatglm3-6b", trust_remote_code=True)
print(model)
