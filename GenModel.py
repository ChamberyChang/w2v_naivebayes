from gensim.models.word2vec import LineSentence, Word2Vec

from Morphological_analysis import Morphological_analysis


class GenModel:
    def __init__(self):
        pass

    def create_w2v_model(self, corpus_path, seg_path, model_path, vec_path):
        print ("Morphological Analysing...")
        # corpus_path, y_data = self.create_data(corpus_path)
        morphological = Morphological_analysis()
        corpus = morphological.data_morphological(corpus_path)
        with open(seg_path,'w',encoding='utf-8') as fW:
            for i in range(len(corpus)):
                fW.write(corpus[i])
                fW.write('\n')

        print ("Training Word2Vec model...")
        w2v_model = Word2Vec(LineSentence(seg_path), vector_size=100, window=5, min_count=1, workers=4)
        w2v_model.save(model_path)
        w2v_model.wv.save_word2vec_format(vec_path, binary=False)
