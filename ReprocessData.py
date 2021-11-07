from Config import Config
from gensim.models import KeyedVectors
from ProcessData import ProcessData
from MorphologicalAnalysis import MorphologicalAnalysis

import sys
import pandas as pd

class ReprocessData:
    def __init__(self, label):
        self.label = label

    def data_reaugment(self, model, x_data_seg, y_data, x_test, reaugment_path):
        print("model data reaugmenting...")

        morph = MorphologicalAnalysis()
        x_data_neologdn = morph.seq_recover_list(x_data_seg)

        exception_vocab = 0
        x_train, y_train = [], []
        y_labels = []
        
        sys.stdout.flush()
        print("=======================Loading Data===============================")
        print("x_train : ", len(x_data_seg))
        print("x_test  : ", len(x_test))
        print("==================================================================")

        for x, y in zip(x_data_neologdn, y_data):
            print("input :", x, "=>", self.label[y])
            x_train.append(morph.basic_to_result(x))
            y_train.append(y)
            y_labels.append(self.label[y])
            cache = []
            for count, text in enumerate(x):
                try:
                    r = model.most_similar(positive=[text])
                    cache.append(r[0][0])
                except :
                    exception_vocab += 1
                    pass
            print("reaugmented :", cache, "=>", self.label[y])
            x_train.append(morph.basic_to_result(cache))
            y_train.append(y)
            y_labels.append(self.label[y])
        pd.DataFrame({'corpus': x_train,'type': y_labels}).to_csv(reaugment_path,header=False,index=False)
        print("\n")
        print("data reaugmenting complete!")

        print("=======================Augmented Data=============================")
        print("x_train : ", len(x_train))
        print("x_test  : ", len(x_test))
        print("exception vocabrary :",exception_vocab)
        print("==================================================================")

        return x_train, y_train

if __name__=="__main__": 
    config = Config().load()
    augment_path = config['augment']['augment_path']
    model_path = config['augment']['model_reaugment_path']
    reaugment_path = config['augment']['reaugment_path']
    test_path = config['data']['test_path']
    label = config['run']['labels']
    dat = ProcessData(label)

    print("model loading start......")
    model = KeyedVectors.load(model_path, mmap='r')
    print("model loading success!")
    x_train_seg, y_train_raw = dat.load_data(augment_path)
    x_test_raw, y_test = dat.load_data(test_path)
    x_train, y_train = ReprocessData(label).data_reaugment(model, x_train_seg, y_train_raw, x_test_raw, reaugment_path)

