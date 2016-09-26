# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午9:39
# @Author  : Luke
# @Software: PyCharm

vocabFile = open("vocab.txt", 'w')
vecFile = open("vec.txt", 'w')

with open("wiki.zh.text.vector", 'r') as ori_file:
    n = 0
    for line in ori_file:
        n += 1
        if n <= 1:
            continue
        # if n > 10:
        #     break
        print n

        line = line.strip()
        vocab,vec = line.split(' ',1)
        vocabFile.writelines(vocab+"\n")
        vecFile.writelines(vec+"\n")
        # print len(vec.split(" "))

vocabFile.close()
vecFile.close()