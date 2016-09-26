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
main_sgd_4:
1. 先将文本转化为ids
2. 对name和addr分别使用相同的W
3. 使用激活函数

main_sgd_3:
1. 通过重复利用session，使得计算速度大幅加快（主要耗时在Graph的build上）
2. 利用全量的wiki验证
3. 每次epoch之前重新shuffle数据
4. 优化读取文件

加载55W条词典速度太慢！！！！！
'''


class config():
    dataSplitRatio = 0.9  # 切分训练集和验证集
    modelNmae = 'weight/' + 'matchModel_1'  # 保存模型路径
    lr = 0.001  # 学习率
    lrMin = 0.0001
    hiddenSize = 20  # 隐藏层
    dropout = 0.9
    l2 = 0.0001  # 正则
    embeddingSize = 32
    iterationKeep = 5

    # debug = True
    debug = False
    if debug:
        useDefinedVocab = False
        iterNum = 5000
        ratio = 10  # Positive trainData numbers vs negative
        batchSize = 4
        epochNum = 5
    else:
        useDefinedVocab = True
        iterNum = 50000
        batchSize = 32
        ratio = 1. / 1
        epochNum = 30


class ShopCompare():
    def __init__(self, dataPath):
        self._conf = config
        self.loadJieba()
        self.loadVecAndVocab()
        self.loadData(dataPath)
        self.lossPre = np.inf

    def loadJieba(self):
        if self._conf.useDefinedVocab:
            jieba.load_userdict("vocab.txt")
        jieba.cut(" ")

    def loadVecAndVocab(self):
        vocab = loadFile2List("vocab.txt", debug=self._conf.debug)
        self.vectors = loadFile2List("vec.txt", lineAsList=True, sep=" ", debug=self._conf.debug, toNumber=True)
        self._conf.embeddingSize = len(self.vectors[0])
        self.vocab2id = dict([(key, val) for (val, key) in enumerate(vocab)])
        self.id2vocab = dict([(self.vocab2id[key], key) for key in self.vocab2id])
        print "Load Vec and Vocab Done!"

    def loadData(self, dataPath):
        positiveData = loadCsv(dataPath)
        length = len(positiveData)
        labels = map(np.float32, list(np.ones(length)))
        data, labels = self.generateNegativeSamples(positiveData, length, labels)
        data, lengths = self.words2ids(data)
        self.shuffle(data, lengths, labels)
        print "Load data done!"

    def words2ids(self, data):
        dataIds = []
        dataLengths = []

        for items in data:
            ids = []
            lengths = []
            for item in items:
                wordL = [word.encode('utf-8') for word in jieba.cut(item)]
                for word in wordL:
                    if word in self.vocab2id:
                        ids.append(self.vocab2id[word])
                    else:
                        ids.append(0)
                lengths.append(len(wordL))
            dataIds.append(ids)
            dataLengths.append(lengths)
        return dataIds, dataLengths

    def shuffle(self, data, lengths, labels):
        assert len(data) == len(labels)
        index = range(len(data))
        np.random.shuffle(index)
        data, lengths, labels = [data[ind] for ind in index], [lengths[ind] for ind in index], [labels[ind] for ind in
                                                                                                index]
        ind = int(self._conf.dataSplitRatio * len(labels))
        self.trainData, self.trainLengths, self.trainLabels = data[:ind], lengths[:ind], labels[:ind]
        self.testData, self.testLengths, self.testLabels = data[ind:], lengths[ind:], labels[ind:]

    def reShuffle(self):
        data = self.trainData[:]
        data.extend(self.testData)
        labels = self.trainLabels[:]
        labels.extend(self.testLabels)
        lengths = self.trainLengths[:]
        lengths.extend(self.testLengths)
        self.shuffle(data, lengths, labels)

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
        labelsN = map(np.float32, list(np.zeros(len(negativeData))))
        labels.extend(labelsN)
        return data, labels

    def addVars(self):
        with tf.variable_scope("Composition"):
            tf.get_variable("W1", shape=[self._conf.hiddenSize, self._conf.embeddingSize])
            tf.get_variable("b1", shape=[self._conf.hiddenSize])
        with tf.variable_scope("Projection"):
            tf.get_variable("W", shape=[1, self._conf.hiddenSize * 2])
            tf.get_variable("b", shape=[1])

    def addPlaceHolder(self):
        '''
        把一条记录转换为index以后输入模型
        用self.indicesLengthPlaceholder记录每个item的长度
        '''
        self.inputPlaceholder = tf.placeholder(tf.int32, None, 'inputPlaceholder')
        self.indicesLengthPlaceholder = tf.placeholder(tf.int32, (4,), name='indicesLengthPlaceholder')
        self.labelPlaceholder = tf.placeholder(tf.float32, None, 'labelPlaceholder')

    def addEmbedding(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.constant(self.vectors, name="Embedding")

    def createFeedDict(self, j, test=False):
        if not test:
            index = np.random.randint(0, len(self.trainData))
            data = self.trainData[index]
            length = self.trainLengths[index]
            label = self.trainLabels[index]
        else:
            data = self.testData[j]
            length = self.testLengths[j]
            label = self.testLabels[j]
        feed = {self.inputPlaceholder: data,
                self.indicesLengthPlaceholder: length,
                self.labelPlaceholder: label}
        if test:
            return feed, label
        return feed

    def Composition(self):
        with tf.variable_scope("Composition", reuse=True):
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W1))

        windowVec = tf.nn.embedding_lookup(self.embedding, self.inputPlaceholder)
        window = tf.sigmoid(tf.matmul(windowVec, W1, transpose_b=True) + b1)
        index = self.indicesLengthPlaceholder
        h = []
        for i in range(4):
            if i == 0:
                sub_h = tf.reduce_sum(tf.gather(window, tf.range(0, index[0])), reduction_indices=0)
            else:
                sub_h = tf.reduce_sum(
                    tf.gather(window, tf.range(tf.reduce_sum(index[:i]), tf.reduce_sum(index[:i + 1]))),
                    reduction_indices=0)
            h.append(sub_h)
        return tf.sigmoid(h)

    def forward(self, inputVec):
        with tf.variable_scope("Projection", reuse=True):
            W = tf.get_variable("W")  # , shape=[1, self._conf.hiddenSize])
            b = tf.get_variable("b")  # , shape=[1])
        if self._conf.l2:
            tf.add_to_collection("total_loss", 0.5 * self._conf.l2 * tf.nn.l2_loss(W))

        # logit = tf.matmul(W, tf.reshape(inputVec, (-1, 1))) + b
        logit = tf.matmul(tf.concat(1, [W, W]), tf.reshape(inputVec, (-1, 1))) + b
        logit = tf.squeeze(logit)
        return logit

    def addloss(self, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.labelPlaceholder, 'loss')
        # loss = tf.nn.l2_loss(tf.sigmoid(logits) - self.labelPlaceholder, 'loss')
        tf.add_to_collection("total_loss", tf.reduce_mean(loss))
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss

    def optimizer(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self._conf.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def verify(self, sess, logit):
        correctNum = 0.
        for j in xrange(len(self.testLabels)):
            feed, label = self.createFeedDict(j, test=True)
            y = sess.run(logit, feed_dict=feed)
            y = 1 * (np.tanh(y) > 0)
            if int(y) == int(label):
                correctNum += 1
        precision = (correctNum / (j + 1))
        print "\nTest set correction ratio:%s" % precision
        return precision

    def runEpoch(self, sess, loss, trainOp, saver):
        self.reShuffle()
        lossHistory = []
        for i in xrange(self._conf.iterNum):
            feed = self.createFeedDict(i)
            lossLocal, _ = sess.run([loss, trainOp], feed_dict=feed)
            lossHistory.append(lossLocal)

            if (i + 1) % 1000 == 0:
                sys.stdout.write('\rIter {}/{} : loss = {}'.format(i + 1, self._conf.iterNum, np.mean(lossHistory)))
                sys.stdout.flush()

        return np.mean(lossHistory)

    def train(self, continueTrain=False):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.addPlaceHolder()
            self.addVars()
            self.addEmbedding()
            h = self.Composition()
            logit = self.forward(h)
            loss = self.addloss(logit)
            trainOp = self.optimizer(loss)

            if not continueTrain:
                init = tf.initialize_all_variables()
                sess.run(init)
                saver = tf.train.Saver()
            else:
                saver = tf.train.Saver()
                saver.restore(sess, self._conf.modelNmae)

            for epoch in xrange(self._conf.epochNum):
                print "epoch: %s" % epoch
                print time.asctime()
                print "lr = %s" % self._conf.lr

                meanLoss = self.runEpoch(sess, loss, trainOp, saver)
                precision = self.verify(sess, logit)

                if epoch > 3 and self._conf.lr > self._conf.lrMin and not continueTrain:
                    if abs(self.lossPre - meanLoss) / meanLoss < 0.05 or \
                                    meanLoss > self.lossPre:
                        self._conf.lr /= 10
                    else:
                        self.lossPre = meanLoss

                if not os.path.exists('weight'):
                    os.makedirs('weight')
                saver.save(sess, self._conf.modelNmae)
                print time.asctime()
                print "\n"

                if self._conf.lr <= self._conf.lrMin:
                    self._conf.iterationKeep -= 1
                    if self._conf.iterationKeep < 0:
                        break
            if precision < 0.8:
                self._conf = config
                self.train(continueTrain=True)


if __name__ == '__main__':
    shopCompare = ShopCompare('1coupon_suc.csv')
    shopCompare.train()
