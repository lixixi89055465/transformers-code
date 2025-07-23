# BitFit 实战
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, TrainingArguments, Trainer

# step2 加载数据集
ds = Dataset.load_from_disk('../data/alpaca_data_zh')
print(ds[:3])

# step3 数据集预处理
tokenizer = AutoTokenizer.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh')
print('1' * 100)
print(tokenizer)


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer('\n'.join(['Human: ' + example['instruction'], example['input']]).strip()
                            + '\n\nAssistant: ')
    response = tokenizer(example['output'] + tokenizer.eos_token)
    input_ids = instruction['input_ids'] + response['input_ids']
    attention_mask = instruction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instruction['input_ids']) + response['input_ids']
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_ds)

a = tokenizer.decode(tokenized_ds[1]['input_ids'])
print('1' * 100)
print(a)
b = tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]['labels'])))
print('2' * 100)
print(b)

# step4 创建模型
model = AutoModelForCausalLM.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh', low_cpu_mem_usage=True)

c = sum(param.numel() for param in model.parameters())
print('3' * 100)
print(c)
# BitFit
# 选择模型参数里面的所有 bias 部分
num_param = 0
for name, param in model.named_parameters():
    if 'bias' not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()

print(num_param)
a = num_param / sum(param.numel() for param in model.parameters())
print('4' * 100)
print(a)
# Step5 配置训练参数
args = TrainingArguments(
    output_dir='./chatbot',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)
# Step6 创建训练器
trainner = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# Step7 模型训练
trainner.train()
model = model.cuda()
ipt = tokenizer('Human: {}\n {}'.format("考试有哪些技巧? ", "").strip() + "\n\nAssistant: ",
                return_tensors='pt').to(model.device)
a = tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
print('5' * 100)
print(a)

# step8 模型推理
from transformers import pipeline

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

ipt = 'Human: {}\n {}'.format('考试有哪些技巧? ', '').strip() + '\n\n Assistant'
pipe(ipt, max_length=256, do_sample=True, )
