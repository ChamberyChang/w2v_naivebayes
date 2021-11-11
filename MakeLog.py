import pandas as pd

from sklearn import metrics

class MakeLog:
    def __init__(self, label):
        self.label = label
    
    def evaluation(self, x_data, y_data, predict):
        eval_a = 0
        eval_p = 0
        eval_r = 0
        eval_f1 = 0
        data_count = len(x_data)
        
        c_mat = metrics.confusion_matrix(y_data, predict)
        cm_df = pd.DataFrame(c_mat)
        print(cm_df)

        eval_a = metrics.accuracy_score(y_data, predict)
        eval_p = metrics.precision_score(y_data, predict, average="micro")
        eval_r = metrics.recall_score(y_data, predict, average="micro")
        eval_f1 = metrics.f1_score(y_data, predict, average="micro")

        print("============= evaluation summury ==================")
        print("number of data :", data_count)
        print("accuracy : ", eval_a)
        print("precision :", eval_p)
        print("recall :", eval_r)
        print("f-measure :", eval_f1)
        print("===================================================")
        
        return cm_df, data_count, eval_a, eval_p, eval_r, eval_f1
    
    def eval_to_dataframe(self, train_path, test_path, data_count, eval_a, eval_p, eval_r, eval_f1, alpha):
        eval_df = pd.DataFrame([
                        ["train file", train_path],
                        ["test file" , test_path],
                        ["number of data" , data_count],
                        ["accuracy"  , eval_a],
                        ["precision", eval_p],
                        ["recall" , eval_r],
                        ["f-measure" , eval_f1],
                        ["alpha" , alpha],
        ])
        return eval_df

    def table_to_pd(self, x_data, y_data, predict):
        correct = 0
        tab_df = pd.DataFrame(columns=["x_test", "answer", "predict", "TRUE"])
        for x, y, prec in zip(x_data, y_data, predict):
            answer_list = []
            for data in y:
                answer_list.append(self.label[data])
            if(self.label[prec] in answer_list):
                correct += 1
            pd_buff = pd.Series([x, answer_list, self.label[prec], (self.label[prec] in answer_list)],index=tab_df.columns)
            tab_df = tab_df.append(pd_buff, ignore_index=True)
        
        return tab_df
    
    def log_write(self, file_path, cm_def, eval_def, tab_def):
        cm_def.to_excel(file_path, sheet_name='confusion_matrix')
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
            tab_def.to_excel(writer, sheet_name="result", index=False)
            eval_def.to_excel(writer, sheet_name='evaluation', index=False, header=False)
        print("Generating result complete!")
            