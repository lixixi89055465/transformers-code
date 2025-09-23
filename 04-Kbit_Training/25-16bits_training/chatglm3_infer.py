# -*- coding: utf-8 -*-
# @Time : 2025/9/23 23:10
# @Author : nanji
# @Site : 
# @File : chatglm3_infer.py
# @Software: PyCharm
# @Comment :
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "/home/nanji/workspace/chatglm3-6b", \
    trust_remote_code=True)
print(tokenizer)

model = AutoModel.from_pretrained(
    "/home/nanji/workspace/chatglm3-6b",
    trust_remote_code=True, \
    low_cpu_mem_usage=True, \
    torch_dtype=torch.half,\
    device_map='auto'
)
model.chat(tokenizer,'考试的技巧有那些?',history=[])
print(model.chat)
print(tokenizer.build_chat_input)
tokenizer("<|user|>", add_special_tokens=False)
tokenizer.get_command("<|user|>")
tokenizer.build_chat_input("考试的技巧有哪些？", history=[], role="user")
tokenizer.decode([64790, 64792, 64795, 30910,    13, 30910, 32227, 54530, 33741, 34953,
         31514, 64796])
