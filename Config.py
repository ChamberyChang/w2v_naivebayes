import json 
import os

class Config:

    def __init__(self, filepath=None):
        if filepath:
            self.configpath = filepath
        else:
            self.configpath = "./config.json"

    def load(self):
        if not os.path.exists(self.configpath):
            with open('./config.template.json', 'r', encoding='utf8')as fp:
                data = json.load(fp)
            with open(self.configpath, 'w', encoding='utf8') as json_file:
                json_file.write(json.dumps(data, indent=4, ensure_ascii=False))
            print("A new config file has been generated")
            exit(1)    
        with open(self.configpath, 'r', encoding='utf8')as fp:
            try:
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
