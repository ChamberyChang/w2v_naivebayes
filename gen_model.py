from CreateDataset import CreateDataset


model_path = "./w2v_model/word2vec.gensim.model"
corpus_path = "./dataset/train.csv"
corpus_seg_path = './w2v_model/sentences.txt'
corpus_vec_path = './w2v_model/vectors.txt'

dataset = CreateDataset()
dataset.create_w2v_model(corpus_path, corpus_seg_path, model_path, corpus_vec_path)
