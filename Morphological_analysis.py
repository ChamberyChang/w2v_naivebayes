import MeCab
import neologdn

class Morphological_analysis:
    def __init__(self):
        pass

    def text_to_ward(self, text):
        m = MeCab.Tagger("mecabrc")
        m.parse(" ")

        m_normal = neologdn.normalize(text)
        m_text = m.parse(m_normal)
        basic_word = []

        m_text = m_text.split("\n")
        for row in m_text:
            word = row.split("\t")[0]
            if word == "EOS":
                break
            else:
                pos = row.split("\t")[4].split(",")
                parts = pos[0]
                if "記号" in parts:
                    if word != "。":
                        continue    
                    basic_word.append(word)
                elif "助" in parts:
                    pass
                elif "形容詞" in parts or "動詞" in parts:
                    basic_word.append(word)
                    pass
                elif "名詞" in parts or "副詞" in parts:
                    basic_word.append(word)
                    pass

        result_word = " ".join(basic_word)
        return result_word

    def list_to_ward(self, text):
        m = MeCab.Tagger("mecabrc")
        m.parse(" ")

        m_normal = neologdn.normalize(text)
        m_text = m.parse(m_normal)
        basic_word = []

        m_text = m_text.split("\n")
        for row in m_text:
            word = row.split("\t")[0]
            if word == "EOS":
                break
            else:
                pos = row.split("\t")[4].split(",")
                parts = pos[0]
                if "記号" in parts:
                    if word != "。":
                        continue    
                    basic_word.append(word)
                elif "助" in parts:
                    pass
                elif "形容詞" in parts or "動詞" in parts:
                    basic_word.append(word)
                    pass
                elif "名詞" in parts or "副詞" in parts:
                    basic_word.append(word)
        return basic_word

    def basic_to_result(self, basic):
        result_word = " ".join(basic)
        return result_word

    def data_morphological(self, xtrain):
        x_train = []
        for x in xtrain:
            x_train.append(self.text_to_ward(x))
        return x_train
    
    def train_morphological(self, train):
        train_data = []
        for data in train:
            train_data.append(self.list_to_ward(data))
        
        return train_data