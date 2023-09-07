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
    def __init__(self, input_channel=6, hidden_channel=12, output_channel=6):
        super(NARXModel, self).__init__()
        self.lin = nn.Linear(input_channel, hidden_channel)
        self.lstm = nn.LSTM(2*hidden_channel, hidden_channel)
        self.output_layer = nn.Linear(hidden_channel, output_channel)
        self.time_embed = nn.Embedding(366, 48)
        self.time_lin = nn.Linear(48, hidden_channel)
        
    def forward(self, x, corr_x, t):
        """
        x: (N, 6)
        corr_x: (6, 6)d
        t: (N, 1)
        
        """
        x = F.relu(self.lin(torch.matmul(x, corr_x)))   # (N, H)   
        t = self.time_lin(self.time_embed(t.long()))           # (N, H)
        y = torch.cat([x, t], dim=1)                    # (N, 2H)
        y, _ = self.lstm(y)
        y = y.squeeze(dim=1)
        y = self.output_layer(y)
        return y


class TimeDataSet:
    def __init__(self, data):
        self.dataset = None
        self.data = data
        self.train_size = int(len(self.data) * 0.8)
        self.train_data = self.data[0:self.train_size, :] 
        self.valid_size = int(len(self.data) * 0.9)     
        self.val_data = self.data[0:self.valid_size, :]
        self.test_data = self.data
        
    def get_train_dataset(self, data_size, max_num, save=False):
        train_dataset = list()
        max_num = min(self.train_size - data_size + 1, max_num)
        filename = 'dataset/train_narx_arima_{}_{}.pkl'.format(data_size, max_num)
        if os.path.exists(filename):
            train_dataset = pickle.load(open(filename, "rb"))
            return train_dataset
        for i in range(max_num):
            cur_data = self.train_data[i:i+data_size, :]
            t = torch.tensor(cur_data[:, 0], dtype=torch.float32)
            x = cur_data[:, 1:]
            x_corr = torch.tensor(self.cal_sim_matrix(x), dtype=torch.float32)
            x_pred = torch.tensor(self.arima_pred(x), dtype=torch.float32).T
            x = torch.tensor(x, dtype=torch.float32)
            x_diff = x - x_pred
            train_dataset.append([x, x_corr, t, x_diff, x_pred])
        if save:
            with open(filename,"wb") as f:
                pickle.dump(train_dataset, f)
        return train_dataset
    
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