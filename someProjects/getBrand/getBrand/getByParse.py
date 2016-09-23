# -*- coding: utf-8 -*-
# @Time    : 16-8-21 上午12:12
# @Author  : Luke
# @Software: PyCharm
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from jpype import *

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import util.conf as conf


class resolveByParse():
    def __init__(self, path):
       self.Brands = self.LoadBrandsDict(path)
       startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/luke/data/HanLP/hanlp-1.2.8.jar:/home/luke/data/HanLP",
                "-Xms1g", "-Xmx1g")
       self.HanLP = JClass('com.hankcs.hanlp.HanLP')
       print self.HanLP.segment('你好，欢迎在Python中调用HanLP的API')

    def getBrandFromDict(self, name):
        if name in self.Brands:
            return self.HanLP.segment(name)
            return name
        else:
            raise NotImplementedError

    # def loadUserDict(self, path):
    #     for fileName in os.listdir(path):
    #         with open(path+"/"+fileName, 'r') as file:
    #             for brand in file:

    def LoadBrandsDict(self, path):
        brandDict = {}
        for fileName in os.listdir(path):
            with open(path + "/" + fileName, 'r') as file:
                for brand in file:
                    brand = brand.strip()
                    brandDict[brand] = 1
        return brandDict


if __name__ == '__main__':
    parser = resolveByParse(conf.dictPath + "/BrandDic/")
    name = parser.getBrandFromDict("重庆小面")
    print name
    name = parser.getBrandFromDict("小螺号112")
    print name
