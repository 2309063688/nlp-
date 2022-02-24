import time
import numpy as np
import copy

DATA = 'data/train.conll'
DEV = 'data/dev.conll'
OUT = 'data/myresult.txt'
alpha = 0.1


#-------------参考数据读取与预处理-------------#
#读取数据，预处理，将每句话，对应词性以二元组形式返回
def preprocess(fdata):
    start = 0
    sentence_list = []
    with open(fdata, 'r', encoding='utf-8') as f:
        lines = [line for line in f]
    for i, line in enumerate(lines):
        if len(line) <= 1:
            splits = [[l.split()[1],l.split()[3]] for l in lines[start:i]]
            word_seq = [array[0] for array in splits]
            tag_seq = [array[1] for array in splits]
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start +=1
            sentence_list.append((word_seq, tag_seq))
    return sentence_list


#构建所有词汇，词性的集合
def myset(sentence_list):
    word_seqs = [array[0] for array in sentence_list]
    tag_seqs = [array[1] for array in sentence_list]
    words = sorted(set(word for word_seq in word_seqs for word in word_seq))
    tags = sorted(set(tag for tag_seq in tag_seqs for tag in tag_seq))
    words += ['<UNK>']
    return words, tags


#构建词汇，词性字典
def mydict(words, tags):
    word_dict = {word: i for i, word in enumerate(words)}
    tag_dict = {tag: i for i, tag in enumerate(tags)}
    return word_dict, tag_dict


#-------------训练数据读取与预处理-------------#
#读取数据，将词汇和词性以词汇词性表中下标的二元组形式返回
def train_data_preprocess(fdata, word_dict, tag_dict):
    data = list()
    sentence_list = preprocess(fdata)
    for word_seq, tag_seq in sentence_list:
        word_seq_idx = [word_dict.get(word, word_dict['<UNK>']) for word in word_seq]
        tag_seq_idx = [tag_dict.get(tag) for tag in tag_seq]
        data.append((word_seq_idx, tag_seq_idx))
    return data


#-------------HMM算法实现-------------#
#平滑
def smooth(A, alpha):
    sums = np.sum(A, axis=0)#按列求和
    return (A + alpha) / (sums + len(A) * alpha)


#构造HMM模型的三个参数：状态转移矩阵A，观测状态生成的概率矩阵B，隐藏状态初始概率分布Π
def initial(data, alpha, word_num, tag_num):
    A = np.zeros((tag_num + 1, tag_num + 1))
    B = np.zeros((word_num, tag_num))

    for word_seq_idx, tag_seq_idx in data:
        pre_tag = -1
        for word_idx, tag_idx in zip(word_seq_idx, tag_seq_idx):
            # A[i][j]里保存着从词性i到词性j的数量,句首保存在A[i][-1],句尾保存在A[-1][i]
            A[tag_idx, pre_tag] += 1
            B[word_idx, tag_idx] += 1
            pre_tag = tag_idx
        A[tag_num, pre_tag] += 1
    A = smooth(A, alpha)
    trans = A[:-1, :-1]         #普通迁移概率
    head_trans = A[:-1, -1]     #句首迁移概率
    tail_trans = A[-1, :-1]     #句尾迁移概率

    B = smooth(B, alpha)        #词性生成的概率

    return trans, head_trans, tail_trans, B


#基于维特比算法的HMM，返回最可能的隐藏状态序列
def Viterbi(word_seq_idx, trans, head_trans, tail_trans, B, tag_num):
    length = len(word_seq_idx)
    delta = np.zeros((length, tag_num))
    paths = np.zeros((length, tag_num), dtype='int')


    delta[0] = head_trans * B[word_seq_idx[0]]
    for i in range(1, length):

        temp = trans * delta[i-1]
        paths[i] = np.argmax(temp, axis=1)
        delta[i] = np.max(temp, axis=1) * B[word_seq_idx[i]]

    pre_tag = np.argmax(delta[-1] * tail_trans)

    result = [pre_tag]
    for i in reversed(range(1, length)):
        pre_tag = paths[i, pre_tag]
        result.append(pre_tag)
    result.reverse()
    return result

#-------------任务评价-------------#
def evaluate(data, trans, head_trans, tail_trans, B, tag_num):
    acc_count = 0
    total = 0

    for word_seq_idx, tag_seq_idx in data:
        total += len(word_seq_idx)
        result = Viterbi(word_seq_idx, trans, head_trans, tail_trans, B, tag_num)
        for i,j in enumerate(result):
            if result[i] == tag_seq_idx[i]:
                acc_count += 1
    accuracy = acc_count / total
    return accuracy


sentence_list = preprocess(DATA)
words, tags = myset(sentence_list)
word_dict, tag_dict = mydict(words, tags)
data = train_data_preprocess(DATA, word_dict, tag_dict)
trans, head_trans, tail_trans, B = initial(data, alpha, len(words), len(tags))
dev_data = train_data_preprocess(DEV, word_dict, tag_dict)
accuracy = evaluate(dev_data, trans, head_trans, tail_trans, B, len(tags))

print("正确率为:", accuracy)