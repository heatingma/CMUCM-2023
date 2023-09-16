import pandas as pd
import numpy as np
from model_train import get_time_dataset
from models import LSTMModel, ElasticModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def execute_model():
    # lstm-arima model
    lstm_arima = LSTMModel()
    lstm_arima.fit(get_time_dataset())
    lstm_arima.save_diff_diff()
    
    # get model prediction
    pred, arima_pred = lstm_arima.predict(predict_size=7)
    pred_volume = pred[:, 0:6]
    for i in range(pred_volume.shape[0]):
        for j in range(pred_volume.shape[1]):
            if abs(pred_volume[i][j] - arima_pred[i][j]) > 10 or pred_volume[i][j] < 0:
                pred_volume[i][j] = arima_pred[i][j]
    pred_cost = arima_pred[:, 6:12]
    pred_price = pred[:, 12:18]
    
    # get history data
    df_real_volume = pd.read_excel("processed_data/time_sale_all.xlsx").dropna()
    volume = np.array(df_real_volume)[:, 1:].astype(np.float32)
    df_real_cost = pd.read_excel("processed_data/cost_all.xlsx").dropna()
    cost_price = np.array(df_real_cost)[:, 1:].astype(np.float32)
    cost = cost_price[:, range(0, 11, 2)]
    price = cost_price[:, range(1, 12, 2)]

    # get diff_diff and reselect the history data
    diff_diff = np.load("processed_data/diff_diff.npy", allow_pickle=True)
    # diff_diff = diff_diff * (lstm_arima.diff_max.numpy() - \
    #     lstm_arima.diff_min.numpy()) + lstm_arima.diff_min.numpy()
    diff_diff = diff_diff * lstm_arima.diff_std.numpy() + lstm_arima.diff_mean.numpy()
    length = diff_diff.shape[0]
    volume = volume[-length:, :]
    cost = cost[-length:, :]
    price = price[-length:, :]

    # fit elastic model    
    elastic_model = ElasticModel(volume, cost, price, diff_diff)
    info, benifit = elastic_model.define_next_price(pred_volume, pred_cost, 7)

    
    for i in range(6):
        part_info = info[:, i, :].T
        part_benifit = np.expand_dims(benifit[:, i], axis=0)
        np_output = np.concatenate([part_info, part_benifit], axis=0)
        df_output = pd.DataFrame(data=np_output)
        df = df_output.rename(
            columns={0: '7月1日', 1: '7月2日', 2: '7月3日', 3: '7月4日',
                   4: '7月5日', 5: '7月6日', 6: '7月7日',})
        df["含义"] = np.array(['销量', '成本', '定价', '比率', \
            '预测差值', '销量弹性', '成本弹性', '收益'])
        df.to_excel("processed_data/predict_{}.xlsx".format(i))