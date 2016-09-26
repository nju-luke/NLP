# -*- coding: utf-8 -*-
# Filename: main.py
# Author：hankcs
# Date: 2015/11/26 14:16

from jpype import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/luke/data/HanLP/hanlp-1.2.8.jar:/home/luke/Projects/address_seg", "-Xms1g", "-Xmx1g")

filepath = "test_add.txt"
file_ori =  open(filepath,'r')
file_seg = open(filepath+'_seg','w')

HanLP = JClass('com.hankcs.hanlp.HanLP')
NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
#segment = HanLP.newSegment().enableAllNamedEntityRecognize(True)
#CRFseg = JClass('com.hankcs.hanlp.seg.CRF.CRFSegment')

CoreSynonymDictionary = JClass('com.hankcs.hanlp.dictionary.CoreSynonymDictionary')

for line in file_ori.readlines():
    line = line.strip('\n')

#    print CRFseg(line)
    
print '\n'

'''
HanLP = JClass('com.hankcs.hanlp.HanLP')
# 中文分词
print HanLP.segment('你好，欢迎在Python中调用HanLP的API')

# 命名实体识别与词性标注
NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
print NLPTokenizer.segment('中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程')

segment = HanLP.newSegment().enablePartOfSpeechTagging(True)
print segment.seg("随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。")

Nshort = JClass('com.hankcs.hanlp.seg.NShort.NShortSegment')
sseg = Nshort().seg("随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。")
print sseg
'''

#shutdownJVM()