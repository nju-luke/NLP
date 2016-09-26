# -*- coding: utf-8 -*-
# @Time    : 16-8-22 上午9:19
# @Author  : Luke
# @Software: PyCharm
import re

import jieba

from util import conf


class resolveByRules():
    def __init__(self, path):
        self.BrandsRemove = self.loadFile(path + "illegalBrands.txt")
        self.keywordsRemove = self.loadFile(path + "illegalWords.txt")
        self.StartWords = self.loadFile(path + "startDict.txt")
        self.EndWords = self.loadFile(path + "endDict.txt")
        self.OneCharWords = self.loadFile(path + "oneCharName.txt")

    def getBrand(self, name):
        name = self.reBraceRemove(name)  ##去括号
        name = self.markRemove(name)  ##去逗号

        name = self.endWithDian(name)

        raise NotImplementedError


        name, flag = self.lookUpDict(name)


        return name

    def reBraceRemove(self, name):
        bracePattern = re.compile(r'(\(|\[|（|【).*(\)|\]|）|】)')
        match = bracePattern.search(name)
        if match:
            name = name.replace(match.group(), "")
        return name

    def markRemove(self, name):
        marks = [",", "+"]
        for mark in marks:
            if mark in name:
                name = name[:name.index(mark)]
        if "-" in name:
            nameList = name.split("-")
            if nameList[1].endswith("店"):
                name = nameList[0]
        return name

    def lookUpDict(self, name):
        raise NotImplementedError

    def endWithDian(self, name):
        name = ""
        dian = "店".decode("utf-8")
        if name.endswith(dian):
            nameList = jieba.cut(name)
            nameList = [na for na in nameList]
            rep = nameList[-1]
            if len(nameList[-1]) == 1:
                rep = nameList[-2]+nameList[-1]
        name = name[:name.index(rep)]
        return name


        # def loadFile(self, path):
        #     words = []
        #     file = open(path, 'r')
        #     for name in file.readlines():
        #         words.append(name.strip())
        #     file.close()
        #     return words


if __name__ == '__main__':
    parser = resolveByRules(conf.dictPath + "/PostProcessingDic/")
    name = parser.getBrand("重庆小面")
    print name
