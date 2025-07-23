# 导入 gradio
import gradio as gr
# 导入 transformers  相关包
from transformers import *

# 通过 Interface加载 pipeline并启动文本分类服务
gr.Interface.from_pipeline(
    pipeline('text-classification',
             model='/home/nanji/workspace/roberta-base-finetuned-dianping-chinese'
             )
).launch()
