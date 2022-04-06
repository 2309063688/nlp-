from config import *
from crf_model import CRFModel
from datetime import datetime

if __name__ == '__main__':
    start_time = datetime.now()
    if shuffle: print("打乱数据集...")
    print("#" * 10 + "开始训练" + "#" * 10)
    crf = CRFModel(TRAIN, DEV)
    crf.mini_batch_train(epoch, exitor, random_seed,learning_rate,decay_rate,lambd,shuffle)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))
