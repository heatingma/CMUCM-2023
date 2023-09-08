import pandas as pd
import numpy as np
from models import TimeDataSet
from datetime import datetime


def date_to_integer(date):
    base_date = datetime(date.year, 1, 1)
    days_diff = (date - base_date).days
    return days_diff

df = pd.read_excel("processed_data/time_sale_all.xlsx")
df.rename(columns={df.columns[0]: 'time'}, inplace=True)
df['time'] = df['time'].apply(date_to_integer)
df = np.array(df).astype(float)
dataset = TimeDataSet(df)


def get_dataset(mode="train", step=30):
    if mode == "train":
        return dataset.get_train_dataset(step, save=True) 
    elif mode == "test":
        return dataset.get_test_dataset(step, save=True) 
    elif mode == "valid":
        return dataset.get_valid_dataset(step, save=True)
    elif mode == "all":
        return dataset.get_all_dataset(step, save=True)
    else:
        raise ValueError("unvaild input mode") 