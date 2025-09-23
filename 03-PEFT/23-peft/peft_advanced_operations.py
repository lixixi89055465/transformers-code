# -*- coding: utf-8 -*-
# @Time : 2025/9/23 21:26
# @Author : nanji
# @Site : 
# @File : peft_advanced_operations.py
# @Software: PyCharm
# @Comment :
# PEFT 进阶操作
# 1. 自定义模型适配
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel

net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

print(net1)
for name, param in net1.named_parameters():
    print(name)

config = LoraConfig(target_modules=['0'])
model1 = get_peft_model(net1, config)

print("2" * 100)
print(model1)
# 2. 多适配器加载与切换
net2 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
print(net2)
config1 = LoraConfig(target_modules=['0'])
model2 = get_peft_model(net2, config1)
model2.save_pretrained('./loraA')
config2 = LoraConfig(target_modules=['2'])
model2 = get_peft_model(net2, config2)
model2.save_pretrained('./loraB')
net2 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
net2 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
print(net2)
model2 = PeftModel.from_pretrained(net2,
                                   model_id='./loraA/',
                                   adapter_name='loraA')
print(model2)

model2.load_adapter('./loraB/',adapter_name='loraB')
print(model2)
print("3"*100)
print(model2.active_adapter)
model2(torch.arange(0,10).view(1,10).float())
model2.set_adapter('loraB')
model2.active_adapter
model2(torch.arange(0,10).view(1,10).float())
## 3. 禁用适配器
model2.set_adapter('loraA')
model2(torch.arange(0,10).view(1,10).float())
with model2.disable_adapter:
    print(model2(torch.arange(0,10).view(1,10).float()))
