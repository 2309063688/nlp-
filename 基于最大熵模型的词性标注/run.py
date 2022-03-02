from corpus import Corpus
from loglinear import LogLinearModel
import random
from datetime import datetime, timedelta

TRAIN = 'data/train.conll'
DEV = 'data/dev.conll'
file = 'data/model.config'

epochs = 100
batch_size = 50
lmbda = 0.01
interval = 5
eta = 0.5
decay = 0.96

print("Preprocess:")
corpus = Corpus(TRAIN)
print(corpus)

print("Load dataset:")
trainset = corpus.load(TRAIN)
devset = corpus.load(DEV)
print("     size of trainset:" + str(len(trainset)))
print("     size of devset:" + str(len(devset)))

start = datetime.now()

print("Create loglinear model")
llm = LogLinearModel(corpus.nt)

llm.create_fspace(trainset)



print("Use SGD to train")
print("  epochs:" + str(epochs))
print("  batch_size:" + str(batch_size))
print("  interval:" + str(interval))
print("  eta:" + str(eta))
print("  decay:" + str(decay))
print("  lmbda:" + str(lmbda))
print()
llm.SGD(trainset=trainset, devset=devset,file=file, epochs=epochs,
        batch_size=batch_size, interval=interval, eta=eta, decay=decay, lmbda=lmbda)

print("time cost:" + str(datetime.now() - start))