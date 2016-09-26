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
1. 用随机梯度下降,不用batch(batch内存一直增加，需要优化）
2. 将inference挪出iteration，内存问题得到解决，运行速度也得到优化
'''


class config():
    iterNum = 10000
    dataSplitRatio = 0.8  # 切分训练集和验证集
    modelNmae = 'weight/' + 'matchModel_1'  # 保存模型路径

    lr = 0.01  # 学习率
    hiddenSize = 20  # 隐藏层
    dropout = 1
    l2 = None #0.0001  # 正则
    embeddingSize = 32

    # debug = True
    debug = False
    if debug:
        ratio = 10  # Positive trainData numbers vs negative
        batchSize = 4
        epochNum = 5
    else:
        batchSize = 32
        ratio = 1. / 1
        epochNum = 30


class ShopCompare():
    def __init__(self, dataPath):
        self._conf = config
        self.loadVecAndVocab()
        self.loadData(dataPath)
        self.loadJieba()
        self.lossPre = np.inf

    def loadJieba(self):
        jieba.load_userdict("vocab.txt")

    def loadVecAndVocab(self):
        vocab = loadFile2List("vocab.txt", debug=self._conf.debug)
        embedding = loadFile2List("vec.txt", lineAsList=True, sep=" ", debug=self._conf.debug)
        self.vectors = [map(np.float32, row) for row in embedding]
        self._conf.embeddingSize = len(self.vectors[0])
        self.vocab2id = dict([(key, val) for (val, key) in enumerate(vocab)])
        self.id2vocab = dict([(self.vocab2id[key], key) for key in self.vocab2id])

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

    def addVars(self):
        with tf.variable_scope("Composition"):
            tf.get_variable("W1", shape=[self._conf.hiddenSize, self._conf.embeddingSize])
            tf.get_variable("b1", shape=[self._conf.hiddenSize])
        with tf.variable_scope("Projection"):
            tf.get_variable("W", shape=[1, self._conf.hiddenSize])
            tf.get_variable("b", shape=[1])

    def addPlaceHolder(self):
        '''
        把一条记录转换为index以后输入模型
        '''
        self.inputPlaceholder = tf.placeholder(tf.int32, None, 'inputPlaceholder')
        self.lablePlaceholder = tf.placeholder(tf.float32, None, 'inputPlaceholder')

    def addEmbedding(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.constant(self.vectors, name="Embedding")

    def createFeedDict(self, j, test=False):
        if not test:
            index = np.random.randint(0, len(self.data))
            data = self.data[index]
            lable = self.labels[index]
        else:
            # index = np.random.randint(0, len(self.testData))
            data = self.testData[j]
            lable = self.testLabels[j]
        ids = []
        for item in data:
            wordL = [word.encode('utf-8') for word in jieba.cut(item)]
            for word in wordL:
                if word in self.vocab2id:
                    ids.append(self.vocab2id[word])
                else:
                    ids.append(0)
        feed = {self.inputPlaceholder: ids,
                self.lablePlaceholder: lable}
        if test:
            return feed,lable
        return feed

    def Composition(self):
        with tf.variable_scope("Composition", reuse=True):
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W1))

        windowVec = tf.nn.embedding_lookup(self.embedding, self.inputPlaceholder)
        window = tf.nn.dropout(tf.matmul(windowVec, W1, transpose_b=True) + b1, self._conf.dropout)
        h = tf.reduce_sum(tf.tanh(window, name="HiddenLayer"), reduction_indices=0)
        return h

    def forward(self, inputVec):
        with tf.variable_scope("Projection", reuse=True):
            W = tf.get_variable("W")  # , shape=[1, self._conf.hiddenSize])
            b = tf.get_variable("b")  # , shape=[1])
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W))

        logit = tf.matmul(W, tf.reshape(inputVec, (-1, 1))) + b
        # logits = tf.tanh(tf.matmul(W, tf.reshape(inputVec, (-1, self._conf.batchSize))) + b)
        logit = tf.squeeze(logit)
        return logit

    def addloss(self, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lablePlaceholder,'loss')
        # loss = tf.nn.l2_loss(tf.tanh(logits) - self.labelPlaceholder, 'loss')
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
            self.addPlaceHolder()
            self.addVars()
            self.addEmbedding()
            saver = tf.train.Saver()
            saver.restore(sess, self._conf.modelNmae)
            h = self.Composition()
            logit = self.forward(h)

            correctNum = 0.
            for j in xrange(len(self.testLabels)):
                feed,label = self.createFeedDict(j, test=True)
                y = sess.run(logit,feed_dict = feed)
                y = 2*(np.tanh(y)>0)-1
                if int(y) == int(label):
                    correctNum += 1
            print "Test set correction ratio:%s" %(correctNum/(j+1))
            print "\n"

    def runEpoch(self, newModel=False, verbose=True):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.addPlaceHolder()
            self.addVars()
            self.addEmbedding()
            # init = tf.initialize_all_variables()
            # sess.run(init)
            if newModel:
                init = tf.initialize_all_variables()
                sess.run(init)
                saver = tf.train.Saver()
            else:
                saver = tf.train.Saver()
                saver.restore(sess, self._conf.modelNmae)
            h = self.Composition()
            logit = self.forward(h)
            loss = self.addloss(logit)
            trainOp = self.optimizer(loss)

            lossHistory = []

            print time.asctime()
            for i in xrange(self._conf.iterNum):
                feed = self.createFeedDict(i)
                lossLocal, _ = sess.run([loss, trainOp], feed_dict=feed)
                lossHistory.append(lossLocal)

                if i % 100 == 0:
                    sys.stdout.write('\r {}/{} : loss = {}'.format(i, len(self.data), np.mean(lossHistory)))
                    sys.stdout.flush()

            if not os.path.exists('weight'):
                os.makedirs('weight')
            saver.save(sess, self._conf.modelNmae)
            print "\n",time.asctime()

            # if np.mean(lossHistory) > self.lossPre:
            if abs(self.lossPre-np.mean(lossHistory))/np.mean(lossHistory) < 0.05 or \
                    np.mean(lossHistory) > self.lossPre:
                self._conf.lr /= 10
            else:
                self.lossPre = np.mean(lossHistory)
            print self._conf.lr

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
