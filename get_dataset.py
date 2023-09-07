import pandas as pd
import numpy as np
from models import TimeDataSet
from datetime import datetime
import pickle
import os


def date_to_integer(date):
    base_date = datetime(date.year, 1, 1)
    days_diff = (date - base_date).days
    return days_diff

df = pd.read_excel("processed_data/time_sale_all.xlsx")
df.rename(columns={df.columns[0]: 'time'}, inplace=True)
df['time'] = df['time'].apply(date_to_integer)
df = np.array(df).astype(float)
dataset = TimeDataSet(df)


def get_train_dataset(data_size=365, max_num=10):
    filename = 'dataset/train_narx_arima_{}_{}.pkl'.format(data_size, max_num)
    if os.path.exists(filename):
        train_dataset = pickle.load(open(filename, "rb"))
    else:
        train_dataset = dataset.get_train_dataset(data_size, max_num, save=True)
    return train_dataset   