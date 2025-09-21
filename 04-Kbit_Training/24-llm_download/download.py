# -*- coding: utf-8 -*-
# @Time : 2025/9/21 20:52
# @Author : nanji
# @Site : 
# @File : download.py
# @Software: PyCharm
# @Comment :
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(model_id='modelscope/Llama-2-7b-ms', \
                  cache_dir='D:/workspace/Llama-2-7b-ms',
                  ignore_file_pattern='*.bin')
