from MorphologicalAnalysis import MorphologicalAnalysis

import sys

import pandas as pd

class CreateDataset:
    def __init__(self):
        pass

    def create_data(self, train_path):
        x_train_list, y_train_list = [], []
        x_train, y_train = [], []

        df_train = pd.read_csv(train_path, header=None, encoding="utf-8")
        
        x_train_list = df_train.iloc[:,0].values.tolist()
        y_train_list = df_train.iloc[:,1].values.tolist()

        # change here as same as the order of labels
        for x, y in zip(x_train_list, y_train_list):
            x_train.append(str(x))
            if "dokujo-tsushin" in y:
                y_train.append(0)
            elif "it-life-hack" in y:
                y_train.append(1)
            elif "kaden-channel" in y:
                y_train.append(2)
            elif "livedoor-homme" in y:
                y_train.append(3)
            elif "movie-enter" in y:
                y_train.append(4)
            elif "peachy" in y:
                y_train.append(5)
            elif "smax" in y:
                y_train.append(6)
            elif "sports-watch" in y:
                y_train.append(7)
            elif "topic-news" in y:
                y_train.append(8)
            else:
                print("Some data should not exist")
                exit(1)
        return x_train, y_train

    def data_augment_use_word2vec(self, model, train_path, test_path):
        print("model data loading...")

        x_train , y_train = [], []
        load_count = 0
        exception_vocab = 0
        morphological = MorphologicalAnalysis()

        x_data, y_data = self.create_data(train_path)
        x_test, y_test = self.create_data(test_path)

        x_data = morphological.list_morphological(x_data)

        sys.stdout.flush()
        print("=======================Before  Dataset============================")
        print("x_train : ", len(x_data))
        #print("x_train => ", x_data[0])
        print("x_test  : ", len(x_test))
        #print("x_test  => ", x_test[0])
        print("==================================================================")

        for x, y in zip(x_data, y_data):
            print(x)
            x_train.append(morphological.basic_to_result(x))
            y_train.append(y)
            buff = []
            for count, text in enumerate(x):
                try:
                    r = model.most_similar(positive=[text])
                    buff.append(r[0][0])
                except :
                    exception_vocab += 1
                    pass
            print(buff)
            x_train.append(buff)
            y_train.append(buff)

        print("\n")
        print("data augment complete!")
        print("=======================Dataset Summury============================")
        print("x_train : ", len(x_train))
        #print("x_train => ", x_train[0])
        print("x_test  : ", len(x_test))
        #print("x_test  => ", x_test[0])
        print("exception vocabrary :",exception_vocab)
        print("==================================================================")

        return x_train, x_test, y_train, y_test
        