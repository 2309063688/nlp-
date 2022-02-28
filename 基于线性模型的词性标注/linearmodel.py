import pickle
import random
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

class LinearModel(object):
    def __init__(self, nt):
        self.nt = nt

    #创建特征空间
    def create_fspace(self, data):
        self.epsilon = list({f for wiseq, tiseq in data
                             for i, ti in enumerate(tiseq)
                             for f in self.instantialize(wiseq, i)})

        # 特征数量
        self.d = len(self.epsilon)

        #特征索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}

        #权重
        self.W = np.zeros((self.d, self.nt))
        self.V = np.zeros((self.d, self.nt))

    #特征模板
    def instantialize(self, wiseq, idx):
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

        if len(word)==1:
            fv.append(('12', word, prev_char, next_char))

        for i in range(1, len(word)):
            prechar, char = word[i-1], word[i]
            if prev_char == char:
                fv.append(('13', char, 'consecutive'))

        if len(word) <= 4:
            fv.append(('14', word))
            fv.append(('15', word))
        else:
            for i in range(1, 5):
                fv.append(('14', word[: i]))
                fv.append(('15', word[-i:]))

        return fv

    #在线学习
    def online_training(self, trainset, devset, file, epochs, interval):
        total_time = timedelta()

        max_epoch, max_acc = 0, 0.0

        e = 0
        for epoch in range(1,epochs + 1):
            e = epoch
            start = datetime.now()

            #random.shuffle(trainset)

            #保存当前轮次和上次更新的轮次
            self.step = 0
            self.PRESTEPS = np.zeros((self.d, self.nt), dtype='int')

            for batch in trainset:
                self.update(batch)

            #使用V之前更新一次
            self.V += [(self.step - prestep) * w
                       for prestep, w in zip(self.PRESTEPS, self.W)]

            #输出一次epoch的信息
            print("Epoch:" + str(epoch) + "/" + str(epochs))
            acc, total, accuracy = self.evaluate(trainset)
            print("train:"+ str(acc) + "/" + str(total) + " = " + str(accuracy))
            acc, total, accuracy = self.evaluate(devset)
            print("dev:" + str(acc) + "/" + str(total) + " = " + str(accuracy))
            t = datetime.now() - start
            print("time cost:" + str(t))
            total_time += t

            #保存效果最好的模型
            if accuracy > max_acc:
                self.dump(file)
                max_epoch, max_acc = epoch, accuracy
            elif epoch - max_epoch > interval:
                break

        print("max accuracy:" + str(max_acc))
        print("epoch:" + str(max_epoch))
        print("mean time:" + str(total_time/e))

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            l_model = pickle.load(f)
        return l_model


    def update(self, batch):
        wiseq, tiseq = batch

        for i, ti in enumerate(tiseq):
            p_ti = self.predict(wiseq, i)
            if ti != p_ti:#更新权重
                fv = self.instantialize(wiseq, i)
                fiseq = (self.fdict[f] for f in fv if f in self.fdict)
                for fi in fiseq:
                    prev_w, prev_step = self.W[fi, [ti, p_ti]], self.PRESTEPS[fi, [ti,p_ti]]

                    #w更新前，更新累加权重V
                    self.V[fi, [ti,p_ti]] += (self.step - prev_step) * prev_w

                    self.W[fi, [ti, p_ti]] += [1, -1]

                    #更新轮次记录
                    self.PRESTEPS[fi, [ti, p_ti]] = self.step
                self.step += 1

    #将得分最高的词性作为预测词性
    def predict(self, wiseq, idx):
        fv = self.instantialize(wiseq, idx)
        scores = self.f(fv)
        return np.argmax(scores)

    #计算得分
    def f(self, fv):
        fiseq = [self.fdict[f] for f in fv if f in self.fdict]
        scores = self.V[fiseq]
        return np.sum(scores, axis=0)

    #评价函数
    def evaluate(self, data):
        acc, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            p_tiseq = np.array([self.predict(wiseq, i)
                                for i in range(len(wiseq))])
            acc += np.sum(tiseq == p_tiseq)
        accuracy = acc / total
        return acc, total, accuracy
