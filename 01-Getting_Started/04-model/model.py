from transformers import AutoConfig, AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('/home/nanji/workspace/rbt3', force_download=True)

# 离线加载
model = AutoModel.from_pretrained('/home/nanji/workspace/rbt3')

print(model.config)
print('0' * 100)
config = AutoConfig.from_pretrained('/home/nanji/workspace/rbt3')
print(config)
print('1' * 100)
print(config.output_attentions)

from transformers import BertConfig

# 模型调用
sen = '弱小的我也有大梦想!'
tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/rbt3')
inputs = tokenizer(sen, return_tensors='pt')
print('2' * 100)
print(inputs)

# 不带 Model Head 模型调用
model = AutoModel.from_pretrained('/home/nanji/workspace/rbt3')
output = model(**tokenizer(sen, return_tensors='pt'))
print('3' * 100)
print(output)

print('4' * 100)
print(output.last_hidden_state.size())
print('5' * 100)
print(len(inputs['input_ids'][0]))
print('6' * 100)
print(inputs['input_ids'])

# 带 Model Head 的模型调用
from transformers import AutoModelForSequenceClassification

clz_model = AutoModelForSequenceClassification.from_pretrained('/home/nanji/workspace/rbt3')
a = clz_model(**tokenizer(sen, return_tensors='pt'))
print(a)
print('7' * 100)
print(clz_model.config.num_labels)
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer

clz_model = AutoModelForSequenceClassification.from_pretrained('/home/nanji/workspace/rbt3',
                                                               num_labels=10)
print('8' * 100)
print(clz_model(**inputs))

print(clz_model.config.num_labels)
