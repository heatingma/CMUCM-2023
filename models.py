import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import TimeDataSet
from utils import draw_array, draw_arrays

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMModel(nn.Module):
    """
        lstm Model
    """
    def __init__(self, input_channel=18, hidden_channel=18, 
                 output_channel=18, lstm_hidden=50, lstm_layer=2):
        super(LSTMModel, self).__init__()
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
        x: (N, 18)
        corr_x: (18, 18)
        t: (N, 1)
        diff_x: (N, 18)
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

    def fit(self, dataset:TimeDataSet):
        self.dataset = dataset
        self.diff_std = self.dataset.diff_std
        self.diff_mean = self.dataset.diff_mean
        self.diff_min = self.dataset.diff_min
        self.diff_max = self.dataset.diff_max
        self.arima_pred = self.dataset.data_pred[-7:, ].to(device)
    
    def save_diff_diff(self):
        all_dataset = self.get_dataset(mode="all")
        diff_diff = None
        save_diff = None
        for cur_data in all_dataset:
            x, x_corr, t, x_diff, _, y_diff = cur_data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            
            if diff_diff is not None:
                if save_diff is None:
                    next_price = x[-1, 12:18]
                    last_price = x[-2, 12:18]
                    save_beta = ((next_price - last_price) / last_price).unsqueeze(0)
                    save_diff = diff_diff.unsqueeze(0)
                else:      
                    save_diff = torch.cat([save_diff, diff_diff.unsqueeze(0)], dim=0)
                    next_price = x[-1, 12:18]
                    last_price = x[-2, 12:18]
                    beta = (next_price - last_price) / last_price
                    save_beta = torch.cat([save_beta, beta.unsqueeze(0)], dim=0)
            
            pred_diff = self.forward(x, x_corr, t, x_diff)
            diff_diff = (y_diff - pred_diff)[0:6]

        np.save("processed_data/diff_diff.npy", save_diff.cpu().detach().numpy())
        np.save("processed_data/diff_beta.npy", save_beta.cpu().detach().numpy())
            
    def get_dataset(self, mode="train", step=30):
        if mode == "train":
            return self.dataset.get_train_dataset(step, save=True) 
        elif mode == "test":
            return self.dataset.get_test_dataset(step, save=True) 
        elif mode == "valid":
            return self.dataset.get_valid_dataset(step, save=True)
        elif mode == "all":
            return self.dataset.get_all_dataset(step, save=True)
        else:
            raise ValueError("unvaild input mode") 
        
    def predict(self, predict_size=1):
        self.load_state_dict(torch.load("pretrained_weight/lstm_weight_step_30.pt"))
        predict_result = list()
        # begin state
        last_dataset = self.get_dataset(mode="all")[-1]
        x, x_corr, t, x_diff, _, _ = last_dataset
        x = x.to(device)
        x_corr = x_corr.to(device)
        t = t.to(device)
        x_diff = x_diff.to(device)
        for i in range(predict_size):  
            #predict
            pred_diff = self.forward(x, x_corr, t, x_diff)
            # pred_diff = pred_diff * (self.diff_max - self.diff_min) + self.diff_min
            pred_diff = pred_diff * self.diff_std + self.diff_mean
            pred = pred_diff + self.arima_pred[i]
            predict_result.append(pred.cpu().detach().numpy())
            # update state
            x = torch.cat([x[1:, :], pred.unsqueeze(dim=0)], dim=0)
            x_corr = torch.tensor(self.dataset.cal_sim_matrix(x), dtype=torch.float32).to(device)
            t = (t + 1) % 365
            x_diff = torch.cat([x_diff[1:, :], pred_diff.unsqueeze(dim=0)], dim=0)
        return np.array(predict_result), self.arima_pred.cpu().detach().numpy()
  
    
class ElasticModel:
    def __init__(self, volume, cost, price, diff_diff) -> None:
        self.volume = volume
        self.cost = cost
        self.price = price
        self.diff_diff = diff_diff
        self.avg_diff_diff = np.average(abs(self.diff_diff), axis=0)
        self.data_process()
        
    def data_process(self):
        # ratio
        self.ratio = self.price / self.cost
        # volume_elastic
        self.alpha = (self.diff_diff / (self.volume-self.diff_diff))[1:, :]
        # price_elastic
        self.beta = (self.price[1:, :] - self.price[0:-1, :]) / self.price[0:-1, :]
        # last_info
        self.last_info = np.array([ self.volume[-1, :], 
                                    self.cost[-1, :], 
                                    self.price[-1, :], 
                                    self.ratio[-1, :], 
                                    self.diff_diff[-1, :], 
                                    self.alpha[-1, :], 
                                    self.beta[-1, :]]).T
        # fit elastic
        self.fit_elastic()

    def fit_elastic(self):
        self.elasticity = np.empty(shape=(6,))
        for i in range(self.alpha.shape[1]):
            alpha = self.alpha[:, i]
            beta = self.beta[:, i]
            draw_arrays(alpha[:35]*100, beta[:35]*10, label1="volume change percent", 
                        label2="price change percent", save=True, 
                        filename="arrays_alpha_beta_{}.png".format(i),)
            draw_array((alpha/beta)[:40], save=True, 
                        filename="linear_alpha_beta_{}.png".format(i))
            array = alpha / beta
            array = array[array <= 5]
            array = array[array >= -15]
            self.elasticity[i] = np.average(array)
            np.save("processed_data/elasticity.npy", self.elasticity)
            
    def define_next_price(self, pred_volume: np.ndarray, 
            pred_cost: np.ndarray, predict_size=1):
        if pred_volume.ndim == 1:
            pred_volume = np.expand_dims(pred_volume, axis=0)
            pred_cost = np.expand_dims(pred_cost, axis=0)
        
        info_list = list()
        benifit_list = list()
        for i in range(predict_size):
            info, benifit = self._define_next_price(pred_volume[i], pred_cost[i])
            info_list.append(info)
            benifit_list.append(benifit)
            
        return np.array(info_list), np.array(benifit_list)
            
    def _define_next_price(self, pred_volume: np.ndarray, pred_cost: np.ndarray):
        info_list = list()
        benifit_list = list()
        for i in range(len(pred_cost)):
            info, benifit = self.get_max_benifit(pred_volume[i], pred_cost[i], \
                self.last_info[i, :], self.elasticity[i], self.avg_diff_diff[i])
            info_list.append(info)
            benifit_list.append(benifit)
        self.last_info = np.array(info_list)
        
        return np.array(info_list), np.array(benifit_list)
    
    def get_max_benifit(self, pred_volume, pred_cost, last_info, elasticity, avg_diff):
        # gain last data
        [last_volume, last_cost, last_price, last_ratio, \
        last_diff_diff, last_alpha, last_beta] = last_info
        
        # define benifit func
        def benifit_func(volume):
            diff_diff = volume - pred_volume
            alpha = diff_diff / pred_volume
            beta = alpha / elasticity
            price = (beta + 1) * last_price
            benifit = volume * (price - pred_cost)
            return benifit
        
        # step caculate  
        ub = pred_volume*1.05 + min(avg_diff, abs(last_diff_diff))
        lb = pred_volume*0.95 - min(avg_diff, abs(last_diff_diff))
        step_size = (ub - lb) / 100
        best_benifit = -1
        best_volume = pred_volume
        for i in range(100):
            volume = ub + step_size * i
            benifit = benifit_func(volume)
            if benifit > best_benifit:
                best_benifit = benifit
                best_volume = volume

        # info caculate 
        next_volume = best_volume
        next_diff_diff = next_volume - pred_volume
        next_alpha = next_diff_diff / pred_volume
        next_beta = next_alpha / elasticity
        next_price = next_beta * last_price + last_price
        next_cost = pred_cost
        next_ratio = next_price / pred_cost
        
        if next_ratio <= 1.4:
            next_volume = pred_volume
            next_diff_diff = next_volume - pred_volume
            next_alpha = next_diff_diff / pred_volume
            next_beta = next_alpha / elasticity
            next_price = next_beta * last_price + last_price
            next_cost = pred_cost
            next_ratio = next_price / pred_cost
        
        # form next data
        next_info = [next_volume, next_cost, next_price, next_ratio, \
        next_diff_diff, next_alpha, next_beta]
        benifit = next_volume * (next_price -next_cost)
        
        return next_info, benifit




class LSTMDiseaseModel(LSTMModel):
    """
        lstm Model
    """
    def __init__(self):
        super(LSTMDiseaseModel, self).__init__()
        self.disease_lin = nn.Linear(48, self.hidden_channel).to(device)
        self.disease_embed = nn.Embedding(2, 48).to(device)
        self.lstm = nn.LSTM(4*self.hidden_channel, self.lstm_hidden, \
            self.lstm_layer, batch_first=True).to(device)
        
    def forward(self, x:torch.Tensor, corr_x:torch.Tensor, 
                t:torch.Tensor, diff_x: torch.Tensor, disease:torch.Tensor):
        """
        x: (N, 18)
        corr_x: (18, 18)
        t: (N, 1)
        diff_x: (N, 18)
        disease: (N, 1)
        """

        x = F.relu(self.lin(torch.matmul(x, corr_x)))   # (N, H)   
        t = self.time_lin(self.time_embed(t.long()))    # (N, H)
        diff_x = self.diff_lin(diff_x)                  # (N, H)
        disease = self.disease_lin(self.disease_embed(disease.long()))    # (N, H)
        y = torch.cat([x, t, diff_x, disease], dim=1)            # (N, 4H)
        
        # lstm
        h0 = torch.zeros(self.lstm_layer, 1, self.lstm_hidden).to(x.device)
        c0 = torch.zeros(self.lstm_layer, 1, self.lstm_hidden).to(x.device)
        out, _ = self.lstm(y.unsqueeze(dim=0), (h0, c0))
        out = self.output_layer(out[:, -1, :]).squeeze(dim=0)
        return out
        
    def save_diff_diff(self):
        all_dataset = self.get_dataset(mode="all")
        diff_diff = None
        save_diff = None
        for cur_data in all_dataset:
            x, x_corr, t, x_diff, _, y_diff, disease = cur_data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            
            if diff_diff is not None:
                if save_diff is None:
                    next_price = x[-1, 12:18]
                    last_price = x[-2, 12:18]
                    save_beta = ((next_price - last_price) / last_price).unsqueeze(0)
                    save_diff = diff_diff.unsqueeze(0)
                else:      
                    save_diff = torch.cat([save_diff, diff_diff.unsqueeze(0)], dim=0)
                    next_price = x[-1, 12:18]
                    last_price = x[-2, 12:18]
                    beta = (next_price - last_price) / last_price
                    save_beta = torch.cat([save_beta, beta.unsqueeze(0)], dim=0)
            
            pred_diff = self.forward(x, x_corr, t, x_diff)
            diff_diff = (y_diff - pred_diff)[0:6]

        np.save("processed_data/diff_diff.npy", save_diff.cpu().detach().numpy())
        np.save("processed_data/diff_beta.npy", save_beta.cpu().detach().numpy())
            
    def get_dataset(self, mode="train", step=30):
        if mode == "train":
            return self.dataset.get_train_dataset(step, save=True, disease=True) 
        elif mode == "test":
            return self.dataset.get_test_dataset(step, save=True, disease=True) 
        elif mode == "valid":
            return self.dataset.get_valid_dataset(step, save=True, disease=True)
        elif mode == "all":
            return self.dataset.get_all_dataset(step, save=True, disease=True)
        else:
            raise ValueError("unvaild input mode") 
        
    def predict(self, predict_size=1):
        self.load_state_dict(torch.load("pretrained_weight/disease_lstm_weight_step_30.pt"))
        predict_result = list()
        # begin state
        last_dataset = self.get_dataset(mode="all")[-1]
        x, x_corr, t, x_diff, _, _ = last_dataset
        x = x.to(device)
        x_corr = x_corr.to(device)
        t = t.to(device)
        x_diff = x_diff.to(device)
        for i in range(predict_size):  
            #predict
            pred_diff = self.forward(x, x_corr, t, x_diff)
            # pred_diff = pred_diff * (self.diff_max - self.diff_min) + self.diff_min
            pred_diff = pred_diff * self.diff_std + self.diff_mean
            pred = pred_diff + self.arima_pred[i]
            predict_result.append(pred.cpu().detach().numpy())
            # update state
            x = torch.cat([x[1:, :], pred.unsqueeze(dim=0)], dim=0)
            x_corr = torch.tensor(self.dataset.cal_sim_matrix(x), dtype=torch.float32).to(device)
            t = (t + 1) % 365
            x_diff = torch.cat([x_diff[1:, :], pred_diff.unsqueeze(dim=0)], dim=0)
        return np.array(predict_result), self.arima_pred.cpu().detach().numpy()       