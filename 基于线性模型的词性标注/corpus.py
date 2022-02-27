class Corpus(object):
    #参数获取
    def __init__(self, fdata):
        #获取句子
        self.sentences = self.preprocess(fdata)
        #获取所有词汇，词性，字符的集合
        self.words, self.tags, self.chars = self.parse(self.sentences)
        #未知字符
        self.chars += ['<UNK>']

        #获取字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}

        #未知字符索引
        self.ui = self.cdict['<UNK>']


        #数量
        self.ns = len(self.sentences)
        self.nw = len(self.words)
        self.nt = len(self.tags)
        self.nc = len(self.chars)

    #数据预处理
    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r', encoding = 'utf-8') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    ##获取所有词汇，词性，字符的集合
    @staticmethod
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(w for wordseq in wordseqs for w in wordseq))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        chars = sorted(set(''.join(words)))
        return words, tags, chars

    #开发数据集预处理
    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wiseq = [
                tuple(self.cdict.get(c, self.ui) for c in w)
                for w in wordseq]
            tiseq = [self.tdict[t] for t in tagseq]

            data.append((wiseq,tiseq))
        return data

    def __repr__(self):
        info = "corpus("
        info += "num of sentences:" + str(self.ns)
        info += "num of words:" + str(self.nw)
        info += "num of tags:" + str(self.nt)
        info += "num of chars:" + str(self.nc)
        info += ")"
        return info
