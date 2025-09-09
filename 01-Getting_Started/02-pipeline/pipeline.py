from transformers.pipelines import SUPPORTED_TASKS

# 查看 Pipeline支持的任务类型
# for k, v in SUPPORTED_TASKS.items():
#     print(k, v['type'])

# PipeLine 的创建与使用方式
from transformers import *

# 根据任务类型直接创建 Pipeline,默认都是英文的模型
# pipe = pipeline('text-classification')
# pipe('very good!')

# 指定任务类型，再执行模型，创建基于执行模型的 pipeline

# pipe = pipeline(
#     'text-classification',
#     model='/home/nanji/workspace/roberta-base-finetuned-dianping-chinese'
# )
# a = pipe('very good!')
# print('1' * 100)
# print(a)

# 预先加载模型，再闯将 PipeLine
# 这种方式，必须同时指定 model和 tokenizer
# model = AutoModelForSequenceClassification.from_pretrained(
#     '/home/nanji/workspace/roberta-base-finetuned-dianping-chinese')
# tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/roberta-base-finetuned-dianping-chinese')
# pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)
# print('2' * 100)
# print(pipe('我觉得不太行'))
# print('3' * 100)
#
# print(pipe.model.device)
import torch
import time

# times = []
# for i in range(100):
#     torch.cuda.synchronize()
#     start = time.time()
#     pipe('我觉得不太行!')
#     torch.cuda.synchronize()
#     end = time.time()
#     times.append(end - start)
# print(sum(times))
#
# pipe = pipeline(
#     'text-classification',
#     model='/home/nanji/workspace/roberta-base-finetuned-dianping-chinese',
#     device=0
# )
# print('4' * 100)
# print(pipe.model.device)
#
# times = []
# for i in range(100):
#     torch.cuda.synchronize()
#     start = time.time()
#     pipe('我觉得不太行!')
#     torch.cuda.synchronize()
#     end = time.time()
#     times.append(end - start)
# print(sum(times))

# qa_pipe = pipeline('question-answering', model='/home/nanji/workspace/roberta-base-finetuned-dianping-chinese')
# print(qa_pipe)
# QuestionAnsweringPipeline
# a = qa_pipe(question='中国的首都是哪里？', context='中国的首都是北京', max_answer_len=2)
# print(a)

# 其他 Pipeline 示列
# checkpoint = 'google/owlvit-base-patch32'
# checkpoint = '/home/nanji/workspace/owlvit-base-patch32'
# detector = pipeline(model=checkpoint, task='zero-shot-object-detection', device=0)
#
# import requests
# from PIL import Image
#
# url = "./s-well-oj0zeY2Ltk4-unsplash.jpg"
# im = Image.open(url)
# print(im)
#
# predictions = detector(im,
#                        candidate_labels=['hat', 'sunglasses', 'book'],
#                        )
# predictions

from PIL import ImageDraw

from transformers import *
import torch

tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/roberta-base-finetuned-dianping-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/nanji/workspace/roberta-base-finetuned-dianping-chinese")

print(tokenizer)

input_text = '我觉得不太行!'
inputs = tokenizer(input_text, return_tensors="pt")
print(inputs)
res = model(**inputs)
print(res)
logits = res.logits
logits = torch.softmax(logits, dim=-1)
print(logits)

pred = torch.argmax(logits).item()
pred

model.config.id2label
