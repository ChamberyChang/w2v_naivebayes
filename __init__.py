from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from gensim.models.word2vec import Word2Vec

from Morphological_analysis import Morphological_analysis
from CreateDataset import CreateDataset
from MakeLog import MakeLog

import os

#================================================================
# naivebayse initialize
#================================================================
print("=========================================")
print("#         naivebayse initialize         #")
print("=========================================")

train_path = "./dataset/train.csv"
test_path  = "./dataset/test.csv"

dataaugment = False
dataset = CreateDataset()
model_path = "./w2v_model/word2vec.gensim.model"
corpus_path = "./w2v_model/corpus.txt"
corpus_seg_path ='./w2v_model/sentences.txt'
if os.path.exists(model_path):
    # load the file if it has already been trained, to save repeating the slow training step below
    model = Word2Vec.load(model_path)
else:
    # Train Word2Vec model.
    dataset.create_w2v_model(corpus_path, corpus_seg_path, model_path)

#method type
# 0:tfidf
# 1:Bag of Word
# 2:tfidf vector
method = 0

#Evaluation type
# 0:default
# 1:Evaluation method considering similarity
evaluation = 1

alpha = 0.1

label = [
        "test", "test2"]

#================================================================
# generate dataset
#================================================================
print("dataset loading...")
print("dataset augment :", dataaugment)
morphological = Morphological_analysis()

if dataaugment:
    x_train, xtest, y_train, y_test = dataset.data_augment_use_word2vec(model, train_path, test_path)
else:
    x_data, x_test, y_train, y_test = dataset.create_dataset(train_path, test_path)
    x_train = morphological.data_morphological(x_data)
    xtest = morphological.data_morphological(x_test)
print("dataset load complete.")
#================================================================
# evaluate
#================================================================
if evaluation == 1:
    print("Use a special evaluation method.")
    xx_test, yy_test = [], []
    test_size = len(xtest)
    index = -1

    for count in range(0, test_size):
        if xtest[count] in xx_test:
            yy_test[index].append(y_test[count])
        else:
            index += 1
            yy_test.append([])
            yy_test[index].append(y_test[count])
            xx_test.append(xtest[count])
    xtest = xx_test
    y_test = yy_test
else:
    print("Use the usual evaluation method.")

#================================================================
# Naivebayse
#================================================================
corpus = x_train + xtest
train_size = len(x_train)

cv = CountVectorizer()
wc = cv.fit_transform(corpus)
ttf = TfidfTransformer(use_idf = False, sublinear_tf = True)
tfidf = ttf.fit_transform(wc)

if method == 0:
    print("tfidf method...")
    x_train = tfidf[:train_size,:]
    x_test = tfidf[train_size:,:]
elif method == 1:
    print("Countvector(Bag of word) method...")
    x_train = wc[:train_size, :]
    x_test = wc[train_size:, :]
elif method == 2:
    print("tfidf vector method...")
    tfidf_vect = TfidfVectorizer()
    X_tfidf = tfidf_vect.fit_transform(corpus)
    x_train = X_tfidf[:train_size, :]
    x_test = X_tfidf[train_size:, :]

print("Multinomial Naivebayse use...")
clf = MultinomialNB(alpha=alpha, class_prior=None, fit_prior=True)
clf.fit(x_train, y_train)
predict = clf.predict(x_test)
#================================================================
# report log
#================================================================
log = MakeLog(label)
if evaluation == 1:
    ans_df, data_count, correct = log.evaluation_to_pd(xtest, y_test, predict)
    set_df = log.setdata_evaluation_to_pd(train_path, test_path, data_count, correct, evaluation, method, alpha, dataaugment)
else:
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print("==================== summury =====================")
    print("train accracy : ", train_acc)
    print("test  accracy : ", test_acc)
    print("===================================================")
    ans_df = log.history_to_pd(xtest, y_test, predict)
    set_df = log.setting_to_pd(train_path, test_path, train_acc, test_acc, evaluation, method, alpha, dataaugment)
log.log_write("result/result_report.xlsx", ans_df, set_df)
