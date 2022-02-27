from corpus import Corpus
from linearmodel import LinearModel

TRAIN = 'data/train.conll'
DEV = 'data/dev.conll'
file = 'data/model.config'
epochs = 20
interval = 5

print("Preprocess:")
corpus = Corpus(TRAIN)
print(corpus)

print("Load dataset:")
trainset = corpus.load(TRAIN)
devset = corpus.load(DEV)
print("     size of trainset:" + str(len(trainset)))
print("     size of devset:" + str(len(devset)))

print("Create linear model")
lm = LinearModel(corpus.nt)

print("create the feature space")
lm.create_fspace(trainset)

print("Use online-training to train")
print("     epochs:" + str(epochs))
print("  interval:" + str(interval))
lm.online_training(trainset=trainset, devset=devset,
                   file=file, epochs=epochs, interval=interval)

