from gensim.models.word2vec import LineSentence, Word2Vec

from MorphologicalAnalysis import MorphologicalAnalysis

#from ProcessData import ProcessData
from Config import Config


class GenModel:
    def __init__(self):
        pass

    def create_w2v_model(self, model_path, config_file=None):
        config = Config(config_file).load()
        corpus_path = config['models']['corpus_path']
        seg_path = config['models']['corpus_seg_out']
        vec_path = config['models']['corpus_vec_out']
        print ("Morphological Analysing...")
        # corpus_path, category_data = ProcessData.load_data(corpus_path)
        morph = MorphologicalAnalysis()
        corpus = morph.data_morph(corpus_path)
        with open(seg_path,'w',encoding='utf-8') as fW:
            for i in range(len(corpus)):
                fW.write(corpus[i])
                fW.write('\n')

        print ("Training Word2Vec model...")
        w2v_model = Word2Vec(LineSentence(seg_path), vector_size=100, window=5, min_count=1, workers=4)
        w2v_model.save(model_path)
        w2v_model.wv.save_word2vec_format(vec_path, binary=False)

if __name__=="__main__": 
    config = Config().load()
    model_path = config['augment']['model_path']
    GenModel.create_w2v_model(model_path)
