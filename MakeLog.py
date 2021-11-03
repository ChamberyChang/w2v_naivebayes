import pandas as pd

class MakeLog:
    def __init__(self, label):
        self.label = label
    
    def decode(self, number):
        return self.label[number]

    def history_to_pd(self, x_data, y_data, predict):
        ans_df = pd.DataFrame(columns=["x_test", "answer", "predict", "TRUE"])
        for x, y, prec in zip(x_data, y_data, predict):
            pd_buff = pd.Series([x, self.decode(y), self.decode(prec), (self.decode(y) == self.decode(prec))],index=ans_df.columns)
            ans_df = ans_df.append(pd_buff, ignore_index=True)

        return ans_df
    
    def evaluation_to_pd(self, x_data, y_data, predict):
        correct = 0
        data_count = len(x_data)
        ans_df = pd.DataFrame(columns=["x_test", "answer", "predict", "TRUE"])
        for x, y, prec in zip(x_data, y_data, predict):
            answer_list = []
            for data in y:
                answer_list.append(self.decode(data))
            if(self.decode(prec) in answer_list):
                correct += 1
            pd_buff = pd.Series([x, answer_list, self.decode(prec), (self.decode(prec) in answer_list)],index=ans_df.columns)
            ans_df = ans_df.append(pd_buff, ignore_index=True)
        print("============= evaluation summury ==================")
        print("number of data :", data_count)
        print("number of correct answers :", correct)
        print("accuracy : ", correct / data_count)
        print("===================================================")
        
        return ans_df, data_count, correct


    def setting_to_pd(self, train_path, test_path, train_accracy, test_accuracy, evaluation, method, alpha, dataaugment):
        setting_df = pd.DataFrame([
                                    ["train_path", train_path],
                                    ["test_path" , test_path],
                                    ["train_acc" , train_accracy],
                                    ["test_acc"  , test_accuracy],
                                    ["evaluation", evaluation],
                                    ["method"    , method],
                                    ["alpha"     , alpha],
                                    ["dataaugment", dataaugment]
        ])
        return setting_df
    
    def setdata_evaluation_to_pd(self, train_path, test_path, data_count, correct, evaluation, method, alpha, dataaugment):
        setting_df = pd.DataFrame([
                                    ["train_path", train_path],
                                    ["test_path" , test_path],
                                    ["data_count" , data_count],
                                    ["correct"  , correct],
                                    ["accuracy", correct / data_count],
                                    ["evaluation", evaluation],
                                    ["method"    , method],
                                    ["alpha"     , alpha],
                                    ["dataaugment", dataaugment]
        ])
        return setting_df
    
    def log_write(self, file_path, ans_def, set_def):
        set_def.to_excel(file_path, sheet_name='set_data')
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
            ans_def.to_excel(writer, sheet_name="result", index=False)