# -*- coding: utf-8 -*-
# @Time : 2025/9/17 20:48
# @Author : nanji
# @Site : 
# @File : summarization_glm.py
# @Software: PyCharm
# @Comment :
# 基于GLM的文本摘要
## Step1 导入相关包
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

## Step2 加载数据集
ds = Dataset.load_from_disk('./nlpcc_2017/')
# print(ds)

ds = ds.train_test_split(100, seed=42)
# print(ds)
# print(ds['train'][0])

## Step3 数据处理
# 对于高版本的Transformers加载会报错，需要修改源码
# 文件地址 ~/.cache\huggingface\modules\transformers_modules\
# THUDM\glm-large-chinese\230f54e413fab4bc8f29bd3508aab301d757ef3e\tokenization_glm.py
# 231行 super().__init__(**kwargs) 移动至 235行，
# 放至self.sp_model.Load(vocab_file)的后面一行
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/glm-large-chinese', trust_remote_code=True)
# print(tokenizer)
contents = ['摘要生成: \n' + e + tokenizer.mask_token for e in ds['train']['content'][:10]]
inputs = tokenizer(contents, truncation=True, max_length=384, padding='max_length', return_tensors='pt')
inputs1 = tokenizer.build_inputs_for_generation(inputs, targets=ds['train']['title'][:10], padding=True, max_gen_length=64)


def process_func(examples):
    contents = ['摘要生成: \n' + e + tokenizer.mask_token for e in examples['content']]
    inputs = tokenizer(contents, truncation=True, max_length=384, padding='max_length', return_tensors='pt')
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=examples['title'], padding=True, max_gen_length=64)
    return inputs


tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds["train"].column_names)
# print(tokenized_ds)

# a = tokenizer.decode(tokenized_ds["train"][0]["input_ids"])
tokenized_ds["train"][0]["labels"]
# print(tokenized_ds["train"][0]["position_ids"])
# Step4 创建模型
model = AutoModelForSeq2SeqLM.from_pretrained('/home/nanji/workspace/glm-large-chinese', trust_remote_code=True)
## Step6 配置训练参数
args = Seq2SeqTrainingArguments(
    output_dir='./summary_glm',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=8,
    num_train_epochs=1
)
# Step7 创建训练器
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds["train"],
    tokenizer=tokenizer,
)
# Step8 模型训练
# trainer.train()
## Step9 模型推理
input_text = ds['test'][-1]['content']
inputs = tokenizer("摘要生成:\n" + input_text + tokenizer.mask_token, return_tensors="pt")
inputs = inputs.to('cuda')
output = model.generate(**inputs, max_new_tokens=64, eos_token_id=tokenizer.eop_token_id, do_sample=True)

t1 = tokenizer.decode(output[0].tolist())
# print(t1)
import torch

model = model.eval()

def predict_test():
    predict = []
    with torch.inference_mode():
        for d in ds["test"]:
            inputs = tokenizer("摘要生成: \n" + d["content"] + tokenizer.mask_token, return_tensors="pt")
            inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=64)
            inputs = inputs.to("cuda")
            output = model.generate(**inputs, max_new_tokens=64, eos_token_id=tokenizer.eop_token_id, do_sample=True)
            predict.append(tokenizer.decode(output[0].tolist()).split("<|startofpiece|>")[1].replace("<|endofpiece|>", "").strip())
            print("curID:", len(predict))
    return predict

result = predict_test()
from rouge_chinese import Rouge

rouge = Rouge()

docode_preds = [" ".join(p) for p in result]
decode_labels = [" ".join(l) for l in ds["test"]["title"]]
scores = rouge.get_scores(docode_preds, decode_labels, avg=True)
{
    "rouge-1": scores["rouge-1"]["f"],
    "rouge-2": scores["rouge-2"]["f"],
    "rouge-l": scores["rouge-l"]["f"],
}