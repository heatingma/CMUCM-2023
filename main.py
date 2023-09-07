import pandas as pd
import numpy as np
from utils import get_categorys
from models import arima, TimeDataSet, NARXModel
from datetime import datetime
import pickle
from get_dataset import get_train_dataset
import torch.nn as nn
import torch
from tqdm import trange

def problem_1():
    data = get_categorys()
    data.get_time_sale_info()


def date_to_integer(date):
    base_date = datetime(date.year, 1, 1)
    days_diff = (date - base_date).days
    return days_diff

def problem_2():
    model = NARXModel()
    train_dataset = get_train_dataset(max_num=30)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    model.load_state_dict(torch.load("pretrained_weight/narx_weight.pt"))
    train_epochs = trange(len(train_dataset), ascii=True, leave=True, desc="Epoch", position=0)
    for data in train_dataset:
        x, x_corr, t, x_diff, _ = data
        optimizer.zero_grad()
        pred_diff = model(x, x_corr, t)
        loss = loss_func(pred_diff, x_diff)
        loss.backward()
        optimizer.step()
        train_epochs.set_description("Epoch (Loss=%g)" % round(loss.item(), 5))
    torch.save(model.state_dict(), "pretrained_weight/narx_weight.pt")
        
        
if __name__ == "__main__":
    # problem_1()
    for i in range(1000):
        problem_2()