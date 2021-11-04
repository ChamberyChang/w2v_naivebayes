from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from gensim import models

from MorphologicalAnalysis import MorphologicalAnalysis
from CreateDataset import CreateDataset
from MakeLog import MakeLog
from GenModel import GenModel

import os

#================================================================
# naivebayse initialize
#================================================================
print("=========================================")
print("#         naivebayse initialize         #")
print("=========================================")

train_path = "./data/train.csv"
test_path  = "./data/test.csv"

dataaugment = True
dataset = CreateDataset()
model_path = "./models/chive-1.2-mc90.kv"

corpus_path = "./models/corpus.txt"
corpus_seg_path ='./models/sentences.txt'
corpus_vec_path = './models/vectors.txt'

#method type
# 0:tfidf
# 1:Bag of Word
# 2:tfidf vector
method = 0

alpha = 0.1

label = [
    "独女通信",
    "ITライフハック",
    "家電チャンネル",
    "livedoor HOMME",
    "MOVIE ENTER",
    "Peachy",
    "エスマックス",
    "Sports Watch",
    "トピックニュース"
]

#================================================================
# generate dataset
#================================================================
print("dataset loading...")
print("dataset augment :", dataaugment)
morphological = MorphologicalAnalysis()

if dataaugment:
    if os.path.exists(model_path):
        # load the file if it has already been trained, to save repeating the slow training step below
        # model = models.Word2Vec.load(model_path)
        model = models.KeyedVectors.load(model_path)
    else:
        # Train Word2Vec model.
        gen_model = GenModel()
        gen_model.create_w2v_model(corpus_path, corpus_seg_path, model_path, corpus_vec_path)
    x_train, xtest, y_train, y_test = dataset.data_augment_use_word2vec(model, train_path, test_path)
else:
    x_data, y_train = dataset.create_data(train_path)
    x_test_d, y_test = dataset.create_data(test_path)
    x_train = morphological.data_morphological(x_data)
    xtest = morphological.data_morphological(x_test_d)
print("dataset load complete.")
#================================================================
# alignation
#================================================================

print("Use a coffusion matrix evaluation method.")
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
ytest = yy_test
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
estimator = MultinomialNB(alpha=alpha, class_prior=None, fit_prior=True)
estimator.fit(x_train, y_train)
predict = estimator.predict(x_test)
#================================================================
# report log
#================================================================
log = MakeLog(label)
ans_df, data_count, eval_a, eval_p, eval_r, eval_f1 = log.evaluation(xtest, ytest, predict)
set_df = log.eval_to_dataframe(train_path, test_path, data_count, eval_a, eval_p, eval_r, eval_f1, alpha)
log.log_write("result/result.xlsx", ans_df, set_df)
