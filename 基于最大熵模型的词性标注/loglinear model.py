import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.special import logsumexp

class LogLinearModel(object):
    def __init__(self, nt):
        self.nt = nt

    def create_fspace(self, data):
        # 特征空间
        self.epsilon = list({
            f for wiseq, tiseq in data
            for i, ti in enumerate(tiseq)
            for f in self.instantialize(wiseq, i, ti)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.d)

    def predict(self, wiseq, index):
        fvs = [self.instantialize(wiseq, index, ti)
               for ti in range(self.nt)]
        scores = np.array([self.f(fv) for fv in fvs])
        return np.argmax(scores)

    def f(self, fv):
        scores = [self.W[self.fdict[f]]
                  for f in fv if f in self.fdict]
        return sum(scores)

    # 特征模板
    def instantialize(self, wiseq, idx, ti):
        word = wiseq[idx]
        prev_word = wiseq[idx - 1] if idx > 0 else '^^'
        next_word = wiseq[idx + 1] if idx < len(wiseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fv = []
        fv.append(('02', ti, word))
        fv.append(('03', ti, prev_word))
        fv.append(('04', ti, next_word))
        fv.append(('05', ti, word, prev_char))
        fv.append(('06', ti, word, next_char))
        fv.append(('07', ti, first_char))
        fv.append(('08', ti, last_char))

        for char in word[1:-1]:
            fv.append(('09', ti, char))
            fv.append(('10', ti, first_char, char))
            fv.append(('11', ti, last_char, char))

        if len(word) == 1:
            fv.append(('12', ti, word, prev_char, next_char))

        for i in range(1, len(word)):
            prechar, char = word[i-1], word[i]
            if prev_char == char:
                fv.append(('13', ti, char, 'consecutive'))

        if len(word) <= 4:
            fv.append(('14', ti, word))
            fv.append(('15', ti, word))
        else:
            for i in range(1, 5):
                fv.append(('14', ti, word[: i]))
                fv.append(('15', ti, word[-i:]))

        return fv

    def SGD(self, trainset, devset, file, epochs, batch_size,
            interval, eta, decay, lmbda, anneal, regularize):
        total_time = timedelta()

        max_epoch, max_acc = 0, 0.0

        for epoch in range(1, epochs + 1):
            






            # 输出一次epoch的信息
            print("Epoch:" + str(epoch) + "/" + str(epochs))
            acc, total, accuracy = self.evaluate(trainset)
            print("train:" + str(acc) + "/" + str(total) + " = " + str(accuracy))
            acc, total, accuracy = self.evaluate(devset)
            print("dev:" + str(acc) + "/" + str(total) + " = " + str(accuracy))
            t = datetime.now() - start
            print("time cost:" + str(t))
            total_time += t

            # 保存效果最好的模型
            if accuracy > max_acc:
                self.dump(file)
                max_epoch, max_acc = epoch, accuracy
            elif epoch - max_epoch > interval:
                break

        print("max accuracy:" + str(max_acc))
        print("epoch:" + str(max_epoch))
        print("mean time:" + str(total_time / e))

    def update(self, batch, lmbda, n, eta):


    # 评价函数
    def evaluate(self, data):
        acc, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            p_tiseq = np.array([self.predict(wiseq, i)
                                for i in range(len(wiseq))])
            acc += np.sum(tiseq == p_tiseq)
        accuracy = acc / total
        return acc, total, accuracy

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            llm = pickle.load(f)
        return llm