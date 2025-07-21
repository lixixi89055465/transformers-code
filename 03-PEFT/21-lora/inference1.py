from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# #  加载基础模型
# model = AutoModelForCausalLM.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh')
tokenizer = AutoTokenizer.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh')
# # 加载 lora模型
# p_model = PeftModel.from_pretrained(
#     model,
#     model_id='./chatbot/checkpoint-500/'
# )
# ipt = tokenizer('Human:{} \n {} '.format('足球好玩吗?', '').strip() + '\n\nAssistant: ', return_tensors='pt')
# tokenizer.decode(p_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)
# # 模型合并
# merge_model = p_model.merge_and_unload()
# print(merge_model)
# ipt = tokenizer(
#     'Human :{} \n {}'.format('足球好玩吗?', '').strip() + '\n\n Assistant:',
#     return_tensors='pt'
# )
#
# tokenizer.decode(p_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)
#
# # 完整模型保存
# merge_model.save_pretrained('./chatbot/merge_model')
#
# merge_model = AutoModelForCausalLM.from_pretrained('/home/sdb2/workspace/bloom-1b4-zh')
model = AutoModelForCausalLM.from_pretrained('./chatbot/merge_model')
# tokenizer = AutoTokenizer.from_pretrained('./chatbot/merge_model')
ipt = tokenizer('Human:{} \n {} '.format('足球好玩吗?', '').strip() + '\n\nAssistant: ', return_tensors='pt')
a = tokenizer.decode(model.generate(**ipt, do_sample=False, max_new_tokens=1000)[0], skip_special_tokens=True)
print(a)
