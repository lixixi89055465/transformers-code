from transformers import AutoTokenizer

sen = '弱小的我也有大梦想'
# Step1 加载与保存
# 从 HuggingFace 加载,输入模型名称，即可加载对于的分词器


# model = AutoModelForSequenceClassification.from_pretrained(
#     '/home/nanji/workspace/roberta-base-finetuned-dianping-chinese')
# tokenizer = AutoTokenizer.from_pretrained('/home/nanji/workspace/roberta-base-finetuned-dianping-chinese')
# print(tokenizer)
# 从HuggingFace加载，输入模型名称，即可加载对于的分词器
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/roberta-base-finetuned-dianping-chinese")
print(tokenizer)
# tokenizer保存到本地
tokenizer.save_pretrained("./roberta_tokenizer")
# 从本地加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('./roberta_tokenizer/')
print(tokenizer)
# step2 句子分词
tokens = tokenizer.tokenize(sen)
print('1' * 100)
# print(tokens)
# step3 查看词典
print('2' * 100)
# print(tokenizer.vocab)
print('3' * 100)
print(tokenizer.vocab_size)
# step4 索引转换
# 将词序列转化为 id 序列
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# 将 id序列转化为 token序列
tokens = tokenizer.convert_ids_to_tokens(ids)
print('4' * 100)
print(tokens)
# 将 token序列转化为 string
print('5' * 100)
str_len = tokenizer.convert_tokens_to_ids(tokens)
print(str_len)
# 更便捷的实现方式
# 将字符串转化为 id序列，又称之为 编码
ids = tokenizer.encode(sen, add_special_tokens=True)
print('6' * 100)
print(ids)

# 将 id序列转化为字符串，又称之为解码
str_len = tokenizer.decode(ids, skip_special_tokens=False)
print('7' * 100)
print(str_len)
# step5填充与截断
# 填充
print('8' * 100)
ids = tokenizer.encode(sen, padding='max_length', max_length=15)
print(ids)
# 截断
ids = tokenizer.encode(sen, max_length=5, truncation=True)
print('9' * 100)
print(ids)
# step6 其他输入部分
ids = tokenizer.encode(sen, padding='max_length', max_length=15)
print('0' * 100)
print(ids)
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
print('1' * 100)
print(ids, attention_mask, token_type_ids)

# step7 快速调用方式
inputs = tokenizer.encode_plus(sen, padding='max_length', max_length=15)
print('2' * 100)
print(inputs)
inputs = tokenizer(sen, padding='max_length', max_length=15)
print(inputs)
# step8 处理 batch 数据
sens = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]

res = tokenizer(sens)
print('3' * 100)
# print(res)
# for i in range(1000):
#     tokenizer(sen)
print('4' * 100)
# print(tokenizer)
# 处理 batch数据
res = tokenizer([sen] * 1000)
# print(tokenizer)
# Fast / Slow Tokenizer
print('5' * 100)
sen = '弱小的我也有大 Dreaming!'
fast_tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/roberta-base-finetuned-dianping-chinese")
print(fast_tokenizer)
print('6' * 100)
slow_tokenizer = AutoTokenizer.from_pretrained(
    "/home/nanji/workspace/roberta-base-finetuned-dianping-chinese",
    use_fast=False
)
print(slow_tokenizer)
# 单循环处理
for i in range(10000):
    slow_tokenizer(sen)

res = fast_tokenizer([sen] * 10000)
# 处理 batch 数据
res = slow_tokenizer([sen] * 10000)
inputs = fast_tokenizer(sen, return_offsets_mapping=True)
print('1' * 100)
print(inputs)
test = inputs.word_ids()
print(test)
print('2' * 100)
# inputs = slow_tokenizer(sen, return_offsets_mapping=True)
# print(input)

# 特殊 Tokenizer 的加载
# 新班班额的 trnasforme > 5.34 ，加载 ThUDM/chatglm 会报错，因此这里替换了天宫的模型
tokenizer = AutoTokenizer.from_pretrained(
    '/home/nanji/workspace/Skywork-13B-base', trust_remote_code=True
)
print(tokenizer)
tokenizer.save_pretrained("skywork_tokenizer")
tokenizer = AutoTokenizer.from_pretrained("skywork_tokenizer", trust_remote_code=True)

a = tokenizer.decode(tokenizer.encode(sen))
print(a)
