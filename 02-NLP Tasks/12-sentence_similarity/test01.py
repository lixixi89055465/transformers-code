# -*- coding: utf-8 -*-
# @Time : 2025/8/31 22:42
# @Author : nanji
# @Site : 
# @File : test01.py
# @Software: PyCharm
# @Comment :

# p = [[0.1], [0.3], [0.7]]
p = [0.1, 0.3, 0.7]
a = [int(i > 0.5) for i in p]
print(a)
