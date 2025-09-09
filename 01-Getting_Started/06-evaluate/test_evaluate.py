import evaluate

# 在2024-01-11的测试中，list_evaluation_modules无法完全显示支持的评估函数，但不影响使用
# 完成的评估函数可以在 https://huggingface.co/evaluate-metric 中查看
# print(evaluate.list_evaluation_modules())
# evaluate.list_evaluation_modules(
#     module_type='comparison',
#     include_community=False,
#     with_details=True
# )

# 加载评估函数
evaluate.load("accuracy")
# print('0'*100)
