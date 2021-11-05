from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from gensim.models import KeyedVectors

from MorphologicalAnalysis import MorphologicalAnalysis
from ProcessData import ProcessData
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
model_path = "./models/chive-1.2-mc90.kv"
augment_path = "./data/augment.csv"

corpus_path = "./models/corpus.txt"
corpus_seg_path ='./models/sentences.txt'
corpus_vec_path = './models/vectors.txt'

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
# generate data
#================================================================
print("loading...")
print("data augment function :", dataaugment)
morph = MorphologicalAnalysis()
dat = ProcessData(label)
if dataaugment:
    if os.path.exists(augment_path):
        print("augmented data found")
        x_train, y_train = dat.load_data(augment_path)
        print("augmented data loading success!")
        x_test_raw, y_test = dat.load_data(test_path)
    elif os.path.exists(model_path):
        print("No augmented data found")
        # load the file if it has already been trained, to save repeating the slow training step below
        # model = KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf8", unicode_errors='ignore')
        print("model loading start......")
        model = KeyedVectors.load(model_path, mmap ='r')
        print("model loading success!")
        x_train_raw, y_train_raw = dat.load_data(train_path)
        x_test_raw, y_test = dat.load_data(test_path)
        x_train, y_train = dat.data_augment(model, x_train_raw, y_train_raw, x_test_raw, augment_path)
    else:
        # Train Word2Vec model.
        gen_model = GenModel()
        gen_model.create_w2v_model(corpus_path, corpus_seg_path, model_path, corpus_vec_path)
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf8", unicode_errors='ignore')
        x_train_raw, y_train_raw = dat.load_data(train_path)
        x_test_raw, y_test = dat.load_data(test_path)
        x_train, y_train = dat.data_augment(model, x_train_raw, y_train_raw, x_test_raw, augment_path)
else:
    x_train_raw, y_train = dat.load_data(train_path)
    x_test_raw, y_test = dat.load_data(test_path)
    x_train = morph.data_morph(x_train_raw)
x_test = morph.data_morph(x_test_raw)
print("data loading complete.")
#================================================================
# alignation
#================================================================

print("Use a coffusion matrix evaluation method.")
xx_test, yy_test = [], []
test_size = len(x_test)
index = -1

for count in range(0, test_size):
    if x_test[count] in xx_test:
        yy_test[index].append(y_test[count])
    else:
        index += 1
        yy_test.append([])
        yy_test[index].append(y_test[count])
        xx_test.append(x_test[count])
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
    print("Use tfidf method...")
    x_train = tfidf[:train_size,:]
    x_test = tfidf[train_size:,:]
elif method == 1:
    print("Use Countvector method...")
    x_train = wc[:train_size, :]
    x_test = wc[train_size:, :]
elif method == 2:
    print("Use tfidf vector method...")
    tfidf_vect = TfidfVectorizer()
    X_tfidf = tfidf_vect.fit_transform(corpus)
    x_train = X_tfidf[:train_size, :]
    x_test = X_tfidf[train_size:, :]

print("Multinomial Naivebayse use...")
estimator = MultinomialNB(alpha=alpha, fit_prior=True, class_prior=None)
estimator.fit(x_train, y_train)
predict = estimator.predict(x_test)
#================================================================
# report log
#================================================================
log = MakeLog(label)
cm_df, data_count, eval_a, eval_p, eval_r, eval_f1 = log.evaluation(xtest, ytest, predict)
eval_df = log.eval_to_dataframe(train_path, test_path, data_count, eval_a, eval_p, eval_r, eval_f1, alpha)
tab_df = log.table_to_pd(xtest, ytest, predict)
print("Generating result...")
log.log_write("result/result.xlsx", cm_df, eval_df, tab_df)
