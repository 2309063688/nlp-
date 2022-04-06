class Corpus(object):
    def __init__(self, fdata):
        self.name = fdata
        self.sentences = self.preprocess(fdata)

        self.words, self.tags, self.chars = self.parse(self.sentences)

        self.chars += ['<UNK>']

        self.wdict = {w:i for i, w in enumerate(self.words)}
        self.tdict = {t:i for i,t in enumerate(self.tags)}
        self.cdict = {c:i for i,c in enumerate(self.chars)}

        self.ui = self.cdict['<UNK>']

        self.ns = len(self.sentences)
        self.nw = len(self.words)
        self.nt = len(self.tags)
        self.nc = len(self.chars)

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r', encoding = 'utf-8') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                temp = [line.split()[1:4:2] for line in lines[start:i]]
                wordseq,tagseq = zip(*temp)

                start = i+1
                while start<len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    @staticmethod
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(w for wordseq in wordseqs for w in wordseq))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        chars = sorted(set(''.join(words)))
        return words, tags, chars

    #开发集数据预处理，序列化
    def data_preprocess(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wiseq = [tuple(self.cdict.get(c, self.ui) for c in word) for word in wordseq]
            tiseq = [self.tdict[t] for t in tagseq]
            data.append((wiseq,tiseq))
        return data

    def __repr__(self):
        info = self.name + "("
        info += "句子数:" + str(self.ns)
        info += "词数" + str(self.nw)
        info += "词性数:" + str(self.nt)
        info += "字数:" + str(self.nc)
        info += ")"
        return info
