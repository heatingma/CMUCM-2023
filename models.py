import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import pickle
import os

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_arima_d(data: pd.DataFrame):
    d = 0
    while True:
        adf_result = adfuller(data["diff_{}".format(d)].dropna())
        p = adf_result[1]
        if p < 0.05:
            return d
        if d >= 2:
            raise ValueError("too many diffs")
        d += 1  
        data["diff_{}".format(d)] = data["diff_{}".format(d-1)].diff(1)
        
        
def get_arima_p_q(data: pd.DataFrame, arima_d):
    pmax = 2
    qmax = 2
    bic_matrix  =  []
    np_data = np.nan_to_num(data["diff_0"], nan=0).astype(float)
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(np_data, order=(p, arima_d, q)).fit().bic) 
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().idxmin()  
    return p, q


def arima(data: np.ndarray, predict_size=1):
    """
        Predict the future data using ARIMA Modle
    """
    # data process
    df = pd.DataFrame(data=data, columns=["diff_0"])
    # define the d
    d = get_arima_d(df)
    # define the p & q
    p, q = get_arima_p_q(df, d)
    # arima model
    model = ARIMA(data.astype(float), order=(p, d, q)).fit()
    # predict 
    predict = model.predict(start=0, end=len(data) + predict_size - 1)
    return predict

    
class NARXModel(nn.Module):
    """
        NARX Model
    """
    def __init__(self, input_channel=6, hidden_channel=6, 
                 output_channel=6, lstm_hidden=50, lstm_layer=2):
        super(NARXModel, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.lstm_hidden = lstm_hidden
        self.lstm_layer = lstm_layer
        self.lin = nn.Linear(input_channel, hidden_channel).to(device)
        self.lstm = nn.LSTM(3*hidden_channel, lstm_hidden, lstm_layer, batch_first=True).to(device)
        self.output_layer = nn.Linear(lstm_hidden, output_channel).to(device)
        self.time_embed = nn.Embedding(366, 48).to(device)
        self.time_lin = nn.Linear(48, hidden_channel).to(device)
        self.diff_lin = nn.Linear(input_channel, hidden_channel).to(device)
        
    def forward(self, x:torch.Tensor, corr_x:torch.Tensor, 
                t:torch.Tensor, diff_x: torch.Tensor):
        """
        x: (N, 6)
        corr_x: (6, 6)
        t: (N, 1)
        diff_x: (N, 6)
        """
        x = F.relu(self.lin(torch.matmul(x, corr_x)))   # (N, H)   
        t = self.time_lin(self.time_embed(t.long()))    # (N, H)
        diff_x = self.diff_lin(diff_x)
        y = torch.cat([x, t, diff_x], dim=1)            # (N, 3H)
        
        # lstm
        h0 = torch.zeros(self.lstm_layer, 1, self.lstm_hidden).to(x.device)
        c0 = torch.zeros(self.lstm_layer, 1, self.lstm_hidden).to(x.device)
        out, _ = self.lstm(y.unsqueeze(dim=0), (h0, c0))
        out = self.output_layer(out[:, -1, :]).squeeze(dim=0)
        return out


class TimeDataSet:
    def __init__(self, data):
        self.dataset = None
        self.data = data
        self.train_size = int(len(self.data) * 0.8)
        self.train_data = self.data[0:self.train_size, :] 
        self.valid_size = int(len(self.data) * 0.9)     
        self.valid_data = self.data[self.train_size:self.valid_size, :]
        self.test_size = len(self.data)
        self.test_data = self.data[self.valid_size:self.test_size, :]
        
        if os.path.exists("dataset/arima_data_pred.pt"):
            self.data_pred = torch.load("dataset/arima_data_pred.pt")
        else:
            self.data_pred = torch.tensor(self.arima_pred(data[:, 1:]), 
                                          dtype=torch.float32).T
            torch.save(self.data_pred, "dataset/arima_data_pred.pt")
        data = torch.tensor(self.data[:, 1:], dtype=torch.float32)
        self.data_diff = data - self.data_pred
        self.diff_mean = torch.mean(self.data_diff)
        self.diff_std = torch.std(self.data_diff)
        self.data_diff = (self.data_diff - self.diff_mean) / self.diff_std
        
    def get_all_dataset(self, step=30, save=False):
        all_dataset = list()
        filename = 'dataset/all_narx_arima_step_{}.pkl'.format(step)
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
            x = torch.tensor(x, dtype=torch.float32)
            all_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(all_dataset, f)
        return all_dataset

       
    def get_train_dataset(self, step=30, save=False):
        train_dataset = list()
        filename = 'dataset/train_narx_arima_step_{}.pkl'.format(step)
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
            x = torch.tensor(x, dtype=torch.float32)
            train_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(train_dataset, f)
        return train_dataset

    def get_valid_dataset(self, step=30, save=False):
        valid_dataset = list()
        filename = 'dataset/valid_narx_arima_step_{}.pkl'.format(step)
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
            x = torch.tensor(x, dtype=torch.float32)
            valid_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(valid_dataset, f)
        return valid_dataset

    def get_test_dataset(self, step=30, save=False):
        test_dataset = list()
        filename = 'dataset/test_narx_arima_step_{}.pkl'.format(step)
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
            x = torch.tensor(x, dtype=torch.float32)
            test_dataset.append([x, x_corr, t, x_diff, x_pred, y_diff])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(test_dataset, f)
        return test_dataset
    
    def arima_pred(self, data):
        x_pred = list()
        for i in range(data.shape[1]):
            x_pred.append(arima(data[:, i], predict_size=0))
        x_pred = np.array(x_pred)
        return x_pred
        
    def cal_sim_matrix(self, array):
        if type(array) != np.ndarray:
            array = np.array(array)
        sim_matrix = np.corrcoef(array, rowvar=False)
        return sim_matrix