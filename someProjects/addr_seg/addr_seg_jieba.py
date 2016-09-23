# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:21:47 2016

@author: luke
"""


import sys
sys.path.append("../")
import jieba
jieba.load_userdict("/home/luke/Projects/address_seg/dic_region.txt")
#import jieba.posseg as pseg

reload(sys)
sys.setdefaultencoding('utf8')

#jieba.add_word('石墨烯')
#jieba.add_word('凱特琳')
#jieba.del_word('自定义词')

filepath = "address_of_dianping_shops.txt"
file_ori =  open(filepath,'r')
file_seg = open(filepath+'_seg','w')

for line in file_ori.readlines():
    line = line.strip('\n')
#    if '(' in line:
#        indg = line.index("(")    
#        line = line[:indg]             #丢弃括号里面的内容
    words = jieba.cut(line,HMM=False)
    words = [w for w in words]
    print(' '.join(words))
    file_seg.writelines(w+' ' for w in words)
    file_seg.writelines('\n')
    
file_ori.close()
file_seg.close()

#
#test_sent = (
#"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
#"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
#"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
#)
#words = jieba.cut(test_sent)
#print('/'.join(words))
