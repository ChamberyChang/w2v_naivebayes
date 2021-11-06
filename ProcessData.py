from MorphologicalAnalysis import MorphologicalAnalysis

import sys

import pandas as pd


class ProcessData:
    def __init__(self, label):
        self.label = label

    def load_data(self, data_path):
        x_data_list, y_data_list = [], []
        x_data, y_data = [], []

        df_data = pd.read_csv(data_path, header=None, encoding="utf-8")
        
        x_data_list = df_data.iloc[:,0].values.tolist()
        y_data_list = df_data.iloc[:,1].values.tolist()

        for x, y in zip(x_data_list, y_data_list):
            x_data.append(str(x))

            for i,e in enumerate(self.label):
                if e in y:
                    y_data.append(i)
                    break
            else:
                print("Some data should not exist")    
                exit(1)
        return x_data, y_data

    def data_augment(self, model, x_data, y_data, x_test, augment_path):
        print("model data augmenting...")

        morph = MorphologicalAnalysis()
        x_data_neologdn = morph.list_morph(x_data)

        exception_vocab = 0
        x_train, y_train = [], []
        y_labels = []
        
        sys.stdout.flush()
        print("=======================Loading Data===============================")
        print("x_train : ", len(x_data))
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
            print("augmented :", cache, "=>", self.label[y])
            x_train.append(morph.basic_to_result(cache))
            y_train.append(y)
            y_labels.append(self.label[y])
        pd.DataFrame({'corpus': x_train,'type': y_labels}).to_csv(augment_path,header=False,index=False)
        print("\n")
        print("data augmenting complete!")

        print("=======================Augmented Data=============================")
        print("x_train : ", len(x_train))
        print("x_test  : ", len(x_test))
        print("exception vocabrary :",exception_vocab)
        print("==================================================================")

        return x_train, y_train
        