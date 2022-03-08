from corpus import Corpus
from globallinearmodel import GlobalLinearModel
from glm_o import GlobalLinearModel_o


TRAIN = 'data/train.conll'
DEV = 'data/dev.conll'
file = 'data/model.config'
epochs = 100
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
glm = GlobalLinearModel_o(corpus.nt)

print("create the feature space")
glm.create_fspace(trainset)

print("Use online-training to train")
print("     epochs:" + str(epochs))
print("  interval:" + str(interval))
glm.online_training(trainset=trainset, devset=devset,
                   file=file, epochs=epochs, interval=interval)

