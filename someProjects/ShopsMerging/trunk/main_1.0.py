# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午10:14
# @Author  : Luke
# @Software: PyCharm
import jieba
import tensorflow as tf
import numpy as np
import os
import sys

import time

from util import *

'''
超参数： 在config中设置


修改算法时：1.修改preProcessing
            2.修改模型
'''


class config():
    dataSplitRatio = 0.8  # 切分训练集和验证集
    modelNmae = 'weight/' + 'matchModel_1'  # 保存模型路径

    lr = 0.02  # 学习率
    hiddenSize = 20  # 隐藏层
    dropout = 0.8
    l2 = 0.0001  # 正则

    debug = True
    # debug = False
    if debug:
        ratio = 10  # Positive trainData numbers vs negative
        batchSize = 4
        epochNum = 5
    else:
        batchSize = 32
        ratio = 2. / 1
        epochNum = 100


class ShopCompare():
    def __init__(self, dataPath):
        self._conf = config
        self.loadVecAndVocab()
        self.loadData(dataPath)
        self.loadJieba()

    def loadJieba(self):
        jieba.load_userdict("vocab.txt")

    def loadVecAndVocab(self):
        vocab = loadFile2List("vocab.txt", debug=self._conf.debug)
        embedding = loadFile2List("vec.txt", lineAsList=True, sep=" ", debug=self._conf.debug)
        self.vectors = [map(np.float32, row) for row in embedding]
        # print tf.nn.embedding_lookup(self.embedding, [0, 1, 2])
        # sess = tf.Session()
        # print sess.run(tf.nn.embedding_lookup(self.embedding, [0, 1, 2]))
        self._conf.embeddingSize = len(self.vectors[0])
        self.vocab2id = dict([(key, val) for (val, key) in enumerate(vocab)])
        self.id2vocab = dict([(self.vocab2id[key], key) for key in self.vocab2id])

    def addEmbedding(self):
        self.embedding = tf.constant(self.vectors, name="Embedding")

    def loadData(self, dataPath):
        positiveData = loadCsv(dataPath)
        length = len(positiveData)
        labels = map(np.float32, list(np.ones(length)))
        data, labels = self.generateNegativeSamples(positiveData, length, labels)
        data, labels = self.shuffle(data, labels)
        ind = int(self._conf.dataSplitRatio * len(labels))
        self.data, self.labels = data[:ind], labels[:ind]
        self.testData, self.testLabels = data[ind:], labels[ind:]

    def shuffle(self, data, labels):
        assert len(data) == len(labels)
        index = range(len(data))
        np.random.shuffle(index)
        return [data[ind] for ind in index], [labels[ind] for ind in index]

    def generateNegativeSamples(self, positiveData, length, labels):
        negNumbers = int(1. / self._conf.ratio * length)
        negativeData = []
        negativeTuples = []
        for i in range(negNumbers):
            ind1 = np.random.randint(0, length)
            ind2 = np.random.randint(0, length)
            while ind1 == ind2 or (ind1, ind2) in negativeTuples:
                ind1 = np.random.randint(0, length)
                ind2 = np.random.randint(0, length)
            negativeTuples.append([ind1, ind2])
            negativeData.append([positiveData[ind1][0], positiveData[ind1][1],
                                 positiveData[ind2][2], positiveData[ind2][3]])
        data = positiveData
        data.extend(negativeData)
        labelsN = map(np.float32, list(-np.ones(len(negativeData))))
        labels.extend(labelsN)
        return data, labels

    def generateBatch(self, index=0, test=False):
        data = []
        labels = []
        if not test:
            data_ori, labels_ori = self.data, self.labels
        else:
            data_ori, labels_ori = self.testData, self.testLabels
        for i in xrange(self._conf.batchSize * index, self._conf.batchSize * (index + 1)):
            data.append(data_ori[i])
            labels.append(labels_ori[i])
        return data, labels

    def addVars(self):
        with tf.variable_scope("Composition"):
            tf.get_variable("W1", shape=[self._conf.hiddenSize, self._conf.embeddingSize])
            tf.get_variable("b1", shape=[self._conf.hiddenSize])
        with tf.variable_scope("Projection"):
            tf.get_variable("W", shape=[1, self._conf.hiddenSize])
            tf.get_variable("b", shape=[1])

    def Composition(self, data):
        vectors = []
        for item in data:
            h = self.model_1(item)
            vectors.append(tf.nn.dropout(h, self._conf.dropout))
            # h += tf.matmul(vector,W1,transpose_b=True)+b1
        # raise NotImplementedError
        return vectors

    def model_1(self, item):
        '''
        这个模型只考虑最简单的输入输出
        将item[i]分词，然后W整合进H，类似于RNN，将4个词条分词后直接输入RNN模型
        '''
        with tf.variable_scope("Composition", reuse=True):
            W1 = tf.get_variable("W1")  # , shape=[self._conf.hiddenSize, self._conf.embeddingSize])
            b1 = tf.get_variable("b1")  # , shape=[self._conf.hiddenSize])
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W1))

        h = tf.zeros(self._conf.hiddenSize)
        for li in item:
            wordL = [word.encode('utf-8') for word in jieba.cut(li)]

            # todo 对于字典中不存在的词，用“，”类似于“unk”去替代
            # todo 或者将这个词继续拆为单个字

            wordL = [word for word in wordL if word in self.vocab2id]

            # print li, wordL
            index = [self.vocab2id[word] for word in wordL]

            if index == []:  # todo 处理未登录词 ===================
                continue

            vector = tf.nn.embedding_lookup(self.embedding, index)
            y1 = tf.nn.dropout(tf.matmul(vector, W1, transpose_b=True) + b1, self._conf.dropout)

            h += tf.reduce_sum(tf.tanh(y1, name="HiddenLayer"), reduction_indices=0)

            # todo 这里的h现在是当成由4部分相加的一个隐藏层，可以处理成一个4部分的concatenate
        return h

    def forward(self, inputVec):
        with tf.variable_scope("Projection", reuse=True):
            W = tf.get_variable("W")  # , shape=[1, self._conf.hiddenSize])
            b = tf.get_variable("b")  # , shape=[1])
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W))
        logits = tf.matmul(W, tf.reshape(inputVec, (-1, self._conf.batchSize))) + b
        # logits = tf.tanh(tf.matmul(W, tf.reshape(inputVec, (-1, self._conf.batchSize))) + b)
        logits = tf.squeeze(logits)
        return logits

    def addloss(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
        # loss = tf.nn.l2_loss(tf.tanh(logits)-trainLabels,'loss')
        tf.add_to_collection("total_loss", tf.reduce_mean(loss))
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss

    def optimizer(self, loss):
        # optimizer = tf.train.AdamOptimizer(self._conf.lr)
        optimizer = tf.train.GradientDescentOptimizer(self._conf.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def verify(self):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.addVars()
            self.addEmbedding()
            saver = tf.train.Saver()
            saver.restore(sess, self._conf.modelNmae)
            correctNum = 0.
            # for j in xrange(int(len(self.testLabels) / self._conf.batchSize)):
            for j in xrange(30):
                data, labels = self.generateBatch(j, test=True)
                inputVec = self.Composition(data)
                logits = self.forward(inputVec)
                y = sess.run(logits)
                y = 2 * (y > 0) - 1
                correctNum += np.sum([y[i] == self.testLabels[i] for i in range(len(y))])
                print j, correctNum
            print correctNum / ((j + 1) * self._conf.batchSize)

    def runEpoch(self, newModel=False, verbose=True):
        lossHistory = []
        # for i in xrange(int(len(self.trainData)/self._conf.batchSize)):
        config = tf.ConfigProto()

        with tf.Graph().as_default(), tf.Session() as sess:
            self.addVars()
            self.addEmbedding()
            # init = tf.initialize_all_variables()
            # sess.run(init)
            if newModel:
                init = tf.initialize_all_variables()
                sess.run(init)
            else:
                saver = tf.train.Saver()
                saver.restore(sess, self._conf.modelNmae)
            # for i in xrange(int(len(self.trainData) / self._conf.batchSize)):
            for i in xrange(2):
                data, labels = self.generateBatch(i)
                inputVec = self.Composition(data)
                logits = self.forward(inputVec)
                loss = self.addloss(logits, labels)
                trainOp = self.optimizer(loss)

                loss, _ = sess.run([loss, trainOp])
                # loss = sess.run(loss)
                lossHistory.append(loss)

                if verbose:
                    sys.stdout.write('\r {}/{} : loss = {}'.format(i * self._conf.batchSize,
                                                        len(self.data), np.mean(lossHistory)))
                    sys.stdout.flush()
                if i % 50 == 10:
                    saver = tf.train.Saver()
                    if not os.path.exists('weight'):
                        os.makedirs('weight')
                    saver.save(sess, self._conf.modelNmae)
        self.verify()

    def train(self, verbose=True):
        for epoch in xrange(self._conf.epochNum):
            print "epoch: %s" % epoch
            if epoch == 0:
                self.runEpoch(newModel=True)
            else:
                self.runEpoch()


if __name__ == '__main__':
    shopCompare = ShopCompare('1coupon_suc.csv')
    shopCompare.train()
