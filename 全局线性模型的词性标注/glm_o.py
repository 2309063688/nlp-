import pickle
import random
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

class GlobalLinearModel_o(object):
    def __init__(self, nt):
        self.nt = nt

    #创建特征空间
    def create_fspace(self, data):
        self.epsilon = list({f for wiseq, tiseq in data
                             for i, ti in enumerate(tiseq[1:], 1)
                             for f in self.instantialize(wiseq, i, tiseq[i-1])}.union(
            {f for wiseq, tiseq in data
             for f in self.instantialize(wiseq,0,-1)}))

        # 特征数量
        self.d = len(self.epsilon)

        #特征索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}

        #权重
        self.W = np.zeros((self.d, self.nt))
        self.V = np.zeros((self.d, self.nt))

        self.BF = [self.bigram(pre_ti) for pre_ti in range(self.nt)]

    def bigram(self,pre_ti):
        return [('01', pre_ti)]

    #特征模板
    def unigram(self, wiseq, idx):
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

    def instantialize(self, wiseq, idx, pre_ti):
        return self.bigram(pre_ti) + self.unigram(wiseq, idx)

    #在线学习
    def online_training(self, trainset, devset, file, epochs, interval):
        total_time = timedelta()

        max_epoch, max_acc = 0, 0.0

        e = 0
        for epoch in range(1,epochs + 1):
            e = epoch
            start = datetime.now()

            random.shuffle(trainset)

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
            llm = pickle.load(f)
        return llm

    def update(self, batch):
        wiseq, tiseq = batch

        p_tiseq = self.predict(wiseq)
        if not np.array_equal(tiseq,p_tiseq):#更新权重
            pre_ti, pre_p_ti = -1, -1
            for i, (ti, p_ti) in enumerate(zip(tiseq, p_tiseq)):
                T_counts = Counter(self.instantialize(wiseq, i, pre_ti))
                T_fi, T_fcounts = map(list,zip(*[(self.fdict[f], T_counts[f])
                                           for f in T_counts if f in self.fdict]))
                prev_w, prev_step = self.W[T_fi, ti], self.PRESTEPS[T_fi, ti]
                #w更新前，更新累加权重V

                self.V[T_fi, ti] += (self.step - prev_step) * prev_w
                self.W[T_fi, ti] += T_fcounts
                #更新轮次记录
                self.PRESTEPS[T_fi, ti] = self.step

                #同上以预测词性再处理一次，权重减去分值
                P_counts = Counter(self.instantialize(wiseq, i, pre_p_ti))
                P_fi, P_fcounts = map(list, zip(*[(self.fdict[f], P_counts[f])
                    for f in P_counts if f in self.fdict]))

                prev_w, prev_step = self.W[P_fi, p_ti], self.PRESTEPS[P_fi, p_ti]

                self.V[P_fi, p_ti] += (self.step - prev_step) * prev_w
                self.W[P_fi, p_ti] -= P_fcounts
                self.PRESTEPS[P_fi, p_ti] = self.step

                pre_ti, pre_p_ti = ti, p_ti

            self.step += 1

    #将得分最高的词性作为预测词性
    def predict(self, wiseq):
        length = len(wiseq)
        delta = np.zeros((length, self.nt))
        paths = np.zeros((length, self.nt), dtype='int')

        bscores = np.array([self.f(bfv) for bfv in self.BF])

        fv = self.instantialize(wiseq, 0, -1)
        #初始概率分布
        delta[0] = self.f(fv)

        for i in range(1, length):
            uscores = self.f(self.unigram(wiseq, i))

            temp = np.transpose(bscores + uscores) + delta[i - 1]
            paths[i] = np.argmax(temp, axis=1)#行里比较，选出最大值索引
            delta[i] = np.max(temp, axis=1)

        pre_tag = np.argmax(delta[-1])

        result = [pre_tag]
        for i in reversed(range(1, length)):
            pre_tag = paths[i, pre_tag]
            result.append(pre_tag)
        result.reverse()
        return result

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
            p_tiseq = np.array([self.predict(wiseq)])
            acc += np.sum(tiseq == p_tiseq)
        accuracy = acc / total
        return acc, total, accuracy
