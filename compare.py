from utils import draw_three_arrays, draw_arrays
import pandas as pd
import numpy as np
from model_train import train_lstm_model, get_time_dataset
from models import LSTMModel, LSTMDiseaseModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def draw_arima_lstm_fact():
    for epoch in range(10):
        train_lstm_model(epochs=10, mode="last", save=True)
        _draw_arima_lstm_fact(epoch*10+10)


def draw_arima_lstm_disease_fact():
    for epoch in range(10):
        train_lstm_model(epochs=10, mode="last", save=True)
        _draw_arima_lstm_disease_fact(epoch*10+10)  



def _draw_arima_lstm_disease_fact(epoch):
# lstm-arima model
    lstm_arima = LSTMDiseaseModel()
    lstm_arima.fit(get_time_dataset())
    lstm_arima.load_state_dict(torch.load("pretrained_weight/disease_lstm_weight_step_30.pt"))
    
    arima_data = lstm_arima.dataset.data_pred[-100:, 0]
    arima_diff = lstm_arima.dataset.data_diff[-100:, 0]
    fact_data = lstm_arima.dataset.real_data[-100:, 0]
    datasets = lstm_arima.dataset.get_all_dataset(disease=True)[-100: ]
    
    model_pred_diff_list = list()
    for dataset in datasets:
        x, x_corr, t, x_diff, x_pred, y_diff, disease = dataset
        x = x.to(device)
        x_corr = x_corr.to(device)
        t = t.to(device)
        x_diff = x_diff.to(device)
        y_diff = y_diff.to(device)
        disease = disease.to(device)
        pred_diff = lstm_arima.forward(x, x_corr, t, x_diff, disease).cpu().detach().numpy()[0]
        model_pred_diff_list.append(pred_diff)
    
    model_pred_diff = np.array(model_pred_diff_list)
    # model_pred_diff = model_pred_diff * (lstm_arima.diff_max.numpy() \
    #     - lstm_arima.diff_min.numpy()) + lstm_arima.diff_min.numpy()
    model_pred_diff *= lstm_arima.diff_std.numpy()
    model_pred_diff += lstm_arima.diff_mean.numpy()
    model_volume = arima_data + model_pred_diff

    draw_arrays(fact_data, arima_data,  label1="fact", label2="arima", 
                save=True, filename="disease_two_volume_lines.png".format(epoch))  
    draw_arrays(fact_data-arima_data,  model_pred_diff, label1="fact_diff", label2="pred_diff", 
                save=True, filename="disease_diff_two_volume_lines_{}.png".format(epoch))  
    draw_three_arrays(fact_data, arima_data, model_volume, 
                label1="fact", label2="arima", label3="lstm-arima",
                save=True, filename="disease_three_volume_lines_{}.png".format(epoch))
    
    
def _draw_arima_lstm_fact(epoch):
    # lstm-arima model
    lstm_arima = LSTMModel()
    lstm_arima.fit(get_time_dataset())
    lstm_arima.load_state_dict(torch.load("pretrained_weight/lstm_weight_step_30.pt"))
    
    arima_data = lstm_arima.dataset.data_pred[-100:, 0]
    arima_diff = lstm_arima.dataset.data_diff[-100:, 0]
    fact_data = lstm_arima.dataset.real_data[-100:, 0]
    datasets = lstm_arima.dataset.get_all_dataset()[-100: ]
    
    model_pred_diff_list = list()
    for dataset in datasets:
        x, x_corr, t, x_diff, x_pred, y_diff = dataset
        x = x.to(device)
        x_corr = x_corr.to(device)
        t = t.to(device)
        x_diff = x_diff.to(device)
        y_diff = y_diff.to(device)
        pred_diff = lstm_arima.forward(x, x_corr, t, x_diff).cpu().detach().numpy()[0]
        model_pred_diff_list.append(pred_diff)
    
    model_pred_diff = np.array(model_pred_diff_list)
    # model_pred_diff = model_pred_diff * (lstm_arima.diff_max.numpy() \
    #     - lstm_arima.diff_min.numpy()) + lstm_arima.diff_min.numpy()
    model_pred_diff *= lstm_arima.diff_std.numpy()
    model_pred_diff += lstm_arima.diff_mean.numpy()
    model_volume = arima_data + model_pred_diff

    draw_arrays(fact_data, arima_data,  label1="fact", label2="arima", 
                save=True, filename="two_volume_lines.png".format(epoch))  
    draw_arrays(fact_data-arima_data,  model_pred_diff, label1="fact_diff", label2="pred_diff", 
                save=True, filename="diff_two_volume_lines_{}.png".format(epoch))  
    draw_three_arrays(fact_data, arima_data, model_volume, 
                label1="fact", label2="arima", label3="lstm-arima",
                save=True, filename="three_volume_lines_{}.png".format(epoch))