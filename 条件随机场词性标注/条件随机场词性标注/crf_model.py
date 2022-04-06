import pickle
import random
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from corpus import Corpus
from config import *

class CRFModel(object):
    def __init__(self, TRAIN, DEV, TEST=None):
        self.corpus = Corpus(TRAIN)
        self.trainset = self.corpus.data_preprocess(TRAIN)
        self.devset = self.corpus.data_preprocess(DEV)
        self.testset = self.corpus.data_preprocess(TEST) if TEST else None

        print(self.corpus)

        self.update_time = 0
        self.nt = self.corpus.nt
        self.tags = self.corpus.tags
        self.tdict = self.corpus.tdict
        self.create_fspace(self.trainset)

    def create_fspace(self, data):
        """
        创建特征空间，序列字典和权重矩阵
        :param data: 由（句子，词性）二元组组成的列表，句子词性都已序列化
        :return:
        """

        self.epsilon = list({f for wiseq, tiseq in data
                             for i, ti in enumerate(tiseq[1:], 1)
                             for f in self.instantialize(wiseq, i, tiseq[i - 1])}.union(
            {f for wiseq, tiseq in data
             for f in self.instantialize(wiseq, 0, -1)}))


        self.nf = len(self.epsilon)

        self.fdict = {f: i for i, f in enumerate(self.epsilon)}

        self.BF = [self.bigram(pre_ti) for pre_ti in range(self.nt)]
        self.biscores = np.zeros([self.nt, self.nt])

        self.W = np.zeros([self.nf, self.nt])

    def instantialize(self,wiseq,i,pre_ti):
        """
        根据模板实例化特征向量
        :param wiseq: 句子（已序列化）
        :param i:词下标
        :param pre_ti:前一个词的词性(已序列化)
        :return:
        所有实例化的特征向量组成的列表
        """
        return self.bigram(pre_ti) + self.unigram(wiseq, i)

    def bigram(self,pre_ti):
        """
        根据特征模板生成二元特征
        :param pre_ti:前一个词的词性（已序列化）
        :return:
        二元特征组成的列表
        """
        return [('01', pre_ti)]

    def unigram(self, wiseq, idx):
        """
        根据特征模板生成一元特征
        :param wiseq: 句子（已序列化）
        :param idx: 词下标
        :return:
        一元特征组成的列表
        """

        word = wiseq[idx]
        prev_word = wiseq[idx - 1] if idx > 0 else '\\\\'
        next_word = wiseq[idx + 1] if idx < len(wiseq) - 1 else '//'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fv = []
        fv.append(('02', word))
        fv.append(('03', prev_word))
        fv.append(('04', next_word))
        fv.append(('05', word, prev_char))
        fv.append(('06', word, next_char))
        fv.append(('07', first_char))
        fv.append(('08', last_char))

        for char in word[1:-1]:
            fv.append(('09', char))
            fv.append(('10', first_char, char))
            fv.append(('11', last_char, char))

        if len(word) == 1:
            fv.append(('12', word, prev_char, next_char))

        for i in range(1, len(word)):
            prechar, char = word[i - 1], word[i]
            if prechar == char:
                fv.append(('13', char, 'consecutive'))

        if len(word) <= 4:
            fv.append(('14', word))
            fv.append(('15', word))
        else:
            for i in range(1, 5):
                fv.append(('14', word[: i]))
                fv.append(('15', word[-i:]))

        return fv

    def score(self, fv):
        """
        计算某个句子中某词标注为所有词性的得分
        :param fv: 该词的特征向量
        :return:
        得分列表（行向量，每一个元素都是当前词标注为对应下标词性的得分）
        """

        scores = np.array([self.W[self.fdict[f]]
                           for f in fv if f in self.fdict])
        return np.sum(scores, axis=0)

    def propagate(self, wiseq, mode = 'f'):
        """
        求解某个句子所有可能的词性序列的得分之和(前向，后向传播)
        :param wiseq: 句子（已序列化）
        :param mode: 选择模式，‘f’为前向传播，‘b’为后向传播
        :return:
        scores矩阵，维度为 单词数*词性数，scores[i,j]为单词i标注为词性j的所有部分路径得分的总和取对数
        """
        nw = len(wiseq)
        score_m = np.zeros([nw, self.nt])

        if mode == 'f':
            score_m[0] = self.score(self.instantialize(wiseq, 0, -1)) # score_m[0]维度为 1*词性数 [i]为当前时间步转移到第i种词性的得分
            for i in range(1, nw):
                uniscores = self.score(self.unigram(wiseq,i))
                scores = (self.biscores + uniscores).T + score_m[i-1] # scores[j][i]为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前序列的总得分 scores维度为 词性数*词性数
                score_m[i] = logsumexp(scores, axis=1) # 维度为 1*词性数 为第i个词转移到各个词性的概率
        elif mode == 'b':
            for i in range(nw-2,-1,-1):
                uniscores = self.score(self.unigram(wiseq,i+1))  #第i+1个词发射为各个词性的得分 维度：1*词性数
                scores = self.biscores + uniscores + score_m[i+1]
                score_m[i] = logsumexp(scores, axis=1)
        return score_m

    def viterbi_predict(self,wiseq):
        """
        维特比算法，预测词性序列
        :param wiseq: 句子（已序列化）
        :return:
        预测的词性序列
        """
        nw = len(wiseq)
        dp_m = np.zeros([nw,self.nt]) #状态转移矩阵DP
        paths = np.zeros([nw, self.nt], dtype='int')

        dp_m[0] = self.score(self.instantialize(wiseq, 0 ,-1))
        paths[0] = np.full([self.nt], -1) #用-1填充，第一个词无需回溯

        for i in range(1, nw):
            uniscores = self.score(self.unigram(wiseq, i))
            scores = np.array((self.biscores + uniscores).T + dp_m[i-1])
            paths[i] = np.argmax(scores, axis=1)
            dp_m[i] = np.max(scores,axis=1)

        prev = np.argmax(dp_m[-1])
        result = [prev]

        for i in range(nw-1,0,-1):
            prev = paths[i][prev]
            result.append(prev)

        return result[::-1]

    def gradient_descent(self, batch, n, lr=0.3, lambd=0.01):
        """
        梯度下降法更新模型的权重
        :param batch: 一个batch的样本数据
        :param n:batch总数
        :param lr: 学习率
        :param lambd: L2正则化系数
        :return:
        """
        gradients = defaultdict(float)

        for sentence in batch:

            wiseq,tiseq = sentence

            nw = len(wiseq)

            pre_ti = -1
            for i, tag in enumerate(tiseq):
                fvs = self.instantialize(wiseq,i,pre_ti)
                fvis = [self.fdict[fv] for fv in fvs if fv in self.fdict]
                for fvi in fvis:
                    gradients[fvi, tag] -= 1
                pre_ti = tag

            forward_propagate_m = self.propagate(wiseq,mode='f')
            backward_propagate_m = self.propagate(wiseq,mode='b')
            log_Z = logsumexp(forward_propagate_m[-1])# 句子所有可能的词性序列的得分求和取对数（用后向传播矩阵的首行也可以）

            fvs = self.instantialize(wiseq,0,-1)
            fvis = [self.fdict[fv] for fv in fvs if fv in self.fdict]
            probs = np.exp(self.score(fvs) + backward_propagate_m[0] - log_Z)

            for fvi in fvis:
                gradients[fvi] += probs

            for i in range(1, nw):
                unigrams = self.unigram(wiseq,i)
                unigrams_id = [self.fdict[fv] for fv in unigrams if fv in self.fdict]
                uniscores = self.score(unigrams)
                scores = np.array(self.biscores + uniscores)


                probs = np.exp(forward_propagate_m[i-1][:, None] + scores + backward_propagate_m[i] - log_Z)

                for bigrams, p in zip(self.BF, probs):
                    bigrams_id = [self.fdict[fv] for fv in bigrams if fv in self.fdict]
                    for fi in bigrams_id+unigrams_id:
                        gradients[fi] += p


        self.W *= (1-lr*lambd/n)
        for key, gradient in gradients.items():
            self.W[key] -= lr*gradient


        self.biscores = np.array([self.score(bfv) for bfv in self.BF])

    def evaluate(self, dataset):
        """
        评价函数
        :param dataset:Corpus数据对象
        :return:
        返回总词数，正确预测词数，正确率
        """
        total = 0
        correct = 0
        for i, tup in enumerate(dataset):
            wiseq = tup[0]
            tiseq = tup[1]

            predict_tiseq = self.viterbi_predict(wiseq)

            total += len(tiseq)
            correct += len([i for i in range(len(tiseq)) if tiseq[i] == predict_tiseq[i]])
        accuracy = correct / total
        return total,correct,accuracy

    def mini_batch_train(self,epoch=100,exitor=10,random_seed=0,learning_rate = 0.3,decay_rate=0.6,lambd=0.01,shuffle=True):
        """
        小批量梯度下降进行模型训练
        :param epoch:迭代总轮数
        :param exitor:退出轮数
        :param random_seed:随机种子
        :param learning_rate:学习率
        :param decay_rate:学习率衰减速率（用于模拟退火）
        :param lambd:L2正则化系数
        :param shuffle:是否打乱数据集
        :return:
        """
        random.seed(random_seed)

        global_step = 100000
        max_acc = 0
        max_acc_epoch = 0

        step = 0

        if shuffle:
            random.shuffle(self.trainset)

        batches = [self.trainset[i:i+batch_size] for i in range(0,len(self.trainset), batch_size)]
        n = len(batches)

        for e in range(1,epoch+1):
            print("第{:d}轮开始训练...：".format(e))

            for batch in batches:
                self.gradient_descent(batch,n,learning_rate*decay_rate**(step/global_step),lambd)
                step += 1

            print("训练完成")

            train_total_num, train_correct_num, train_accuracy = self.evaluate(self.trainset)
            print("训练集词数为{:d}，预测正确数为{:d}，正确率为:{:f}".format(train_total_num, train_correct_num, train_accuracy))

            dev_total_num, dev_correct_num, dev_accuracy = self.evaluate(self.devset)
            print("开发集词数为{:d}，预测正确数为{:d}，正确率为:{:f}".format(dev_total_num, dev_correct_num, dev_accuracy))

            if self.testset:
                test_total_num, test_correct_num, test_accuracy = self.evaluate(self.testset)
                print("测试集词数为{:d}，预测正确数为{:d}，正确率为:{:f}".format(test_total_num, test_correct_num, test_accuracy))

            if dev_accuracy > max_acc:
                max_acc_epoch = e
                max_acc = dev_accuracy
            elif e - max_acc_epoch >= exitor:
                print("经过{:d}轮模型正确率无提升，结束训练！最大正确率为第{:d}轮训练后的{:f}".format(exitor, max_acc_epoch, max_acc))
                break
            print()