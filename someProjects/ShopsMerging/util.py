# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午10:16
# @Author  : Luke
# @Software: PyCharm
import csv
import numpy as np

WORDS = ["查看位置"]

def loadFile2List(path, lineAsList=None, sep=None, debug = False, toNumber = False):
    lineList = []
    num = 1
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if lineAsList:
                line = line.split(sep)
            if toNumber:
                line = map(np.float32, line)
            lineList.append(line)
            if debug:
                num+=1
                if num > 100:
                    break
    return lineList


def loadCsv(path, type=1):
    data = []
    csvfile = file(path, 'rb')
    reader = csv.reader(csvfile)
    badDataNum = 0
    for line in reader:
        line = "".join(line)
        for word in WORDS:
            line = line.replace(word,"")
        lineList = line.split("|")

        if type == 1:
            if len(lineList) != 5:
                badDataNum += 1
                continue
                # print line
                # raise AttributeError
            data.append([lineList[0],lineList[1],lineList[3],lineList[4]])
    print "Bad trainData number: %s" %badDataNum
        # raise NotImplementedError
    csvfile.close()
    return data

if __name__ == '__main__':
    loadCsv('1coupon_suc.csv')