# -*- coding: utf-8 -*-
# @Time : 2025/8/31 22:49
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV13P411C7UD?t=611.3
# @File : dual_model.py
# @Software: PyCharm
# @Comment :
# 文本相似度实例
# Step1 导入相关包

from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    Trainer, \
    TrainingArguments
from datasets import load_dataset

# Step2 加载数据集
dataset = load_dataset('json', data_files='./train_pair_1w.json', split='train')
print(dataset)

print(dataset[0])
# Step3 划分数据集
datasets = dataset.train_test_split(test_size=0.2)
# datasets
## Step4 数据集预处理


import torch

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/home/nanji/workspace/chinese-macbert-base",
                                          from_tf=True)


# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

def process_function(examples):
    sentences = []
    labels = []
    for sen1, sen2, label in zip(examples['sentence1'], examples['sentence2'], examples['label']):
        sentences.append(sen1)
        sentences.append(sen2)
        labels.append(1 if int(label) == 1 else -1)
    # input_ids, attention_mask, token_type_ids
    tokenized_examples = tokenizer(sentences, max_length=128, truncation=True, padding=True)
    tokenized_examples = {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    tokenized_examples['labelss'] = labels
    return tokenized_examples


tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
print("0" * 100)
print(tokenized_datasets['train'][0])
# Step5 创建模型
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CosineSimilarity, CosineEmbeddingLoss

class DualModel(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step1 分别获取sentenceA 和 sentenceB的输入
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        # Step2 分别获取sentenceA 和 sentenceB的向量表示
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senA_pooled_output = senA_outputs[1]    # [batch, hidden]

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senB_pooled_output = senB_outputs[1]    # [batch, hidden]

        # step3 计算相似度

        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)    # [batch, ]

        # step4 计算loss

        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)

        output = (cos,)
        return ((loss,) + output) if loss is not None else output


model = DualModel.from_pretrained("/home/nanji/workspace/chinese-macbert-base")
# model = DualModel.from_pretrained("hfl/chinese-macbert-base")

# Step6 创建评估函数
import evaluate

acc_metric = evaluate.load('./metric_accuracy.py')
f1_metric = evaluate.load('./metric_f1.py')


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p > 0.7) for p in predictions]
    labels = [int(l > 0) for l in labels]
    # predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


# Step7 创建TrainingArguments
train_args = TrainingArguments(output_dir="./dual_model",  # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               per_device_eval_batch_size=32,  # 验证时的batch_size
                               logging_steps=10,  # log 打印的频率
                               eval_strategy="epoch",  # 评估策略
                               save_strategy="epoch",  # 保存策略
                               save_total_limit=3,  # 最大保存数
                               learning_rate=2e-5,  # 学习率
                               weight_decay=0.01,  # weight_decay
                               metric_for_best_model="f1",  # 设定评估指标
                               load_best_model_at_end=True)  # 训练完成后加载最优模型
## Step8 创建Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=eval_metric
)
# Step9 模型训练
trainer.train()
## Step10 模型评估
trainer.evaluate(tokenized_datasets['test'])


# Step11 模型预测

class SentenceSimilarityPipeline:
    def __init__(self, model, tokenizer) -> None:
        self.model = model.bert
        self.tokenizer = tokenizer
        self.device = model.device

    def preprocess(self, senA, senB):
        return self.tokenizer([senA, senB],
                              max_length=128,
                              truncation=True,
                              return_tensors="pt",
                              padding=True)

    def predict(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)[1]  # [2,768]

    def postprocess(self, logits):
        cos = CosineSimilarity()(logits[None, 0, :], logits[None, 1, :]).squeeze().cpu().item()
        return cos

    def __call__(self, senA, senB, return_vector=False):
        inputs = self.preprocess(senA, senB)
        logits = self.predict(inputs)
        result = self.postprocess(logits)
        if return_vector:
            return result, logits
        else:
            return result


pipe = SentenceSimilarityPipeline(model, tokenizer)
a = pipe('我喜欢北京', '明天不行', return_vector=True)
print(a)
