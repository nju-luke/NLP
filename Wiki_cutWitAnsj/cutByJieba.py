# -*- coding: utf-8 -*-
# @Time    : 16-9-19 上午11:36
# @Author  : Luke
# @Software: PyCharm
import jieba

filename = "wikizh_chs_all"
# filename = "wiki00_chs"

cutFIle = open(filename+"_cut",'w')

'''
debug 设为False以执行全部重写
'''
debug = False
n = 0

def isPassed(line):
    flag = False
    if line.startswith("<"):
        flag = True
    if len(line.decode('utf-8')) < 30:
        # print line,len(line.decode('utf-8'))
        flag = True
    return flag

with open(filename,'r') as file:
    for line in file:
        if isPassed(line):
            continue
        # print line
        lineCut = " ".join(jieba.cut(line))
        # print lineCut
        cutFIle.writelines(lineCut.encode('utf-8'))
        n+=1
        if n%10000 == 0:
            print "%s  \t lines have been processed!"%n
        if debug:
            if n>10:
                break

cutFIle.close()