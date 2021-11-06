import json 
import os

class Config:

    def __init__(self, filepath=None):
        if filepath:
            self.configpath = filepath
        else:
            self.configpath = "./config.json"

    def store(self, data):
        with open(self.configpath, 'w') as json_file:
            json_file.write(json.dumps(data, indent=4))

    def load(self):
        if not os.path.exists(self.configpath):
            with open(self.configpath, 'w') as json_file:
                data = {"data": {"train_path": "./data/","test_path": "./data/"},"augment":{"dataaugment": False,"model_path": "./models/","augment_path": "./data/augment/"},"models":{"corpus_path": "./models/corpus.txt","corpus_seg_out": "./models/sentences.txt","corpus_vec_out": "./models/vectors.txt"},"run":{"method": 0,"alpha": 0.1,"labels": [],"result_path": "./result/"}}
                json_file.write(json.dumps(data, indent=4))
                print("A new config file has been generated")
                exit(1)    
        with open(self.configpath) as json_file:
            try:
                with open('./config.json','r',encoding='utf8')as fp:
                    data = json.load(fp)
                #print("Configuration loaded", type(data))
            except:
                data = {}
            return data

    def set(self, dataname):
        json_obj = self.load()
        for key in dataname:
            json_obj[key] = dataname[key]
        self.store(json_obj)
        print(json.dumps(json_obj, indent=4))
        
    
if __name__=="__main__": 
    config_file=None
    config = Config(config_file).load()
    print(config['run']['labels'])
