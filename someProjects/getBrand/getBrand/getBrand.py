# -*- coding: utf-8 -*-
# @Time    : 16-8-20 下午11:58
# @Author  : Luke
# @Software: PyCharm

import time


class getBrand():
    def __init__(self, path):
        self.dictPath = path

    def getBrand(self, name):
        brand, flag = self.resolveByParse(name)
        if flag:
            return brand
        brand = self.resolveByRules(name)
        return brand

    def resolveByParse(self, name):
        raise NotImplementedError

    def resolveByRules(self, name):
        raise NotImplementedError


if __name__ == '__main__':
    pass