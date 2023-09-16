from utils import get_categorys

from model_train import train_lstm_model, train_lstm_model_disease

import torch
from compare import draw_arima_lstm_fact, draw_arima_lstm_disease_fact
from model_excute import execute_model
from signal_goods import get_single_predict, get_benifit
from integer_program import lip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data_info():
    data = get_categorys()
    data.get_time_sale_info()
    data.get_cost_info()


if __name__ == "__main__":
    # get_data_info()
    train_lstm_model(epochs=100, mode="part", save=True, save_loss=True)
    # train_lstm_model(epochs=10, mode="all", save=True)
    # train_lstm_model(epochs=200, mode="last", save=True, save_loss=True)
    # draw_arima_lstm_fact()
    # execute_model()
    # get_single_predict()
    # get_benifit()
    # lip()
    # train_lstm_model_disease(epochs=10, mode="part", save=True, save_loss=True)
    # train_lstm_model_disease(epochs=10, mode="all", save=True)
    # train_lstm_model_disease(epochs=200, mode="last", save=True, save_loss=True)
    # draw_arima_lstm_disease_fact()