import torch
import numpy as np
import os
import pickle
from arima import arima

class TimeDataSet:
    def __init__(self, data):
        self.dataset = None
        mask = np.isnan(data).any(axis=1)
        self.data = data[~mask]
        self.train_size = int(len(self.data) * 0.8)
        self.train_data = self.data[0:self.train_size, :] 
        self.valid_size = int(len(self.data) * 0.9)     
        self.valid_data = self.data[self.train_size:self.valid_size, :]
        self.test_size = len(self.data)
        self.test_data = self.data[self.valid_size:self.test_size, :]
        
        if os.path.exists("dataset/arima_data_pred.pt"):
            self.data_pred = torch.load("dataset/arima_data_pred.pt")
        else:
            self.data_pred = torch.tensor(arima(self.data[:, 1:], predict_size=7), 
                                          dtype=torch.float32).T
            torch.save(self.data_pred, "dataset/arima_data_pred.pt")
            
        real_data = torch.tensor(self.data[:, 1:], dtype=torch.float32)
        self.real_data = real_data
        self.data_diff = real_data - self.data_pred[:-7]
        self.diff_std = torch.std(self.data_diff)
        self.diff_mean = torch.mean(self.data_diff)
        self.data_diff = (self.data_diff - self.diff_mean) / self.diff_std
        self.diff_min = torch.min(self.data_diff)
        self.diff_max = torch.max(self.data_diff)
        # self.data_diff = (self.data_diff - self.diff_min) / (self.diff_max - self.diff_min)
        self.disease = torch.empty(self.data_diff.shape[0])
        for i in range(len(self.disease)):
            self.disease[i] = 1 if i >= 800 else 0
        self.disease = self.disease.T
        
    def get_all_dataset(self, step=30, save=False, disease=False):
        all_dataset = list()
        filename = 'dataset/all_lstm_arima_step_{}.pkl'.format(step)
        if disease:
            filename = 'dataset/disease_all_lstm_arima_step_{}.pkl'.format(step)
        if os.path.exists(filename):
            all_dataset = pickle.load(open(filename, "rb"))
            return all_dataset
        for i in range(self.test_size - step):
            begin = i
            end = i + step
            cur_data = self.data[begin:end, :]
            t = torch.tensor(cur_data[:, 0], dtype=torch.float32)
            x = cur_data[:, 1:]
            x_corr = torch.tensor(self.cal_sim_matrix(x), dtype=torch.float32)
            x_pred = self.data_pred[begin:end, :]
            x_diff = self.data_diff[begin:end, :]
            y_diff = self.data_diff[end, :]
            t_disease = self.disease[begin:end]
            x = torch.tensor(x, dtype=torch.float32)
            if disease == False:
                all_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
            else:
                all_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff, t_disease])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(all_dataset, f)
        return all_dataset

    def get_train_dataset(self, step=30, save=False, disease=False):
        train_dataset = list()
        filename = 'dataset/train_lstm_arima_step_{}.pkl'.format(step)
        if disease:
            filename = 'dataset/disease_train_lstm_arima_step_{}.pkl'.format(step)
        if os.path.exists(filename):
            train_dataset = pickle.load(open(filename, "rb"))
            return train_dataset
        for i in range(self.train_size - step):
            begin = i
            end = i + step
            cur_data = self.data[begin:end, :]
            t = torch.tensor(cur_data[:, 0], dtype=torch.float32)
            x = cur_data[:, 1:]
            x_corr = torch.tensor(self.cal_sim_matrix(x), dtype=torch.float32)
            x_pred = self.data_pred[begin:end, :]
            x_diff = self.data_diff[begin:end, :]
            y_diff = self.data_diff[end, :]
            t_disease = self.disease[begin:end]
            x = torch.tensor(x, dtype=torch.float32)
            if disease == False:
                train_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
            else:
                train_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff, t_disease])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(train_dataset, f)
        return train_dataset

    def get_valid_dataset(self, step=30, save=False, disease=False):
        valid_dataset = list()
        filename = 'dataset/valid_lstm_arima_step_{}.pkl'.format(step)
        if disease:
            filename = 'dataset/disease_valid_lstm_arima_step_{}.pkl'.format(step)
        if os.path.exists(filename):
            valid_dataset = pickle.load(open(filename, "rb"))
            return valid_dataset
        for i in range(self.valid_size - step - self.train_size):
            begin = i
            end = i + step
            cur_data = self.data[begin:end, :]
            t = torch.tensor(cur_data[:, 0], dtype=torch.float32)
            x = cur_data[:, 1:]
            x_corr = torch.tensor(self.cal_sim_matrix(x), dtype=torch.float32)
            x_pred = self.data_pred[begin:end, :]
            x_diff = self.data_diff[begin:end, :]
            y_diff = self.data_diff[end, :]
            t_disease = self.disease[begin:end]
            x = torch.tensor(x, dtype=torch.float32)
            if disease == False:
                valid_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
            else:
                valid_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff, t_disease])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(valid_dataset, f)
        return valid_dataset

    def get_test_dataset(self, step=30, save=False, disease=False):
        test_dataset = list()
        filename = 'dataset/test_lstm_arima_step_{}.pkl'.format(step)
        if disease:
            filename = 'dataset/disease_test_lstm_arima_step_{}.pkl'.format(step)
        if os.path.exists(filename):
            test_dataset = pickle.load(open(filename, "rb"))
            return test_dataset
        for i in range(self.test_size - step - self.valid_size):
            begin = i
            end = i + step
            cur_data = self.data[begin:end, :]
            t = torch.tensor(cur_data[:, 0], dtype=torch.float32)
            x = cur_data[:, 1:]
            x_corr = torch.tensor(self.cal_sim_matrix(x), dtype=torch.float32)
            x_pred = self.data_pred[begin:end, :]
            x_diff = self.data_diff[begin:end, :]
            y_diff = self.data_diff[end, :]
            t_disease = self.disease[begin:end]
            x = torch.tensor(x, dtype=torch.float32)
            if disease == False:
                test_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
            else:
                test_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff, t_disease])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(test_dataset, f)
        return test_dataset
    
    def cal_sim_matrix(self, array):
        if type(array) != np.ndarray:
            array = array.cpu().detach().numpy()
        sim_matrix = np.corrcoef(array, rowvar=False)
        return sim_matrix