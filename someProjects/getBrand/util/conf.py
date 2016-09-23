# -*- coding: utf-8 -*-
# @Time    : 16-8-20 下午11:34
# @Author  : Luke
# @Software: PyCharm
import os

server = dict(
    port = 8384
)

parentPath = os.path.abspath(os.path.join(os.getcwd(),".."))

dictPath = parentPath+"/"+"resources"

print dictPath