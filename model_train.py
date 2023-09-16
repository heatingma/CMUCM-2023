import pandas as pd
import numpy as np
from models import TimeDataSet, LSTMModel, LSTMDiseaseModel
from datetime import datetime
import torch
from tqdm import trange
import torch.nn as nn
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils import draw_arrays, draw_array

def date_to_integer(date):
    base_date = datetime(date.year, 1, 1)
    days_diff = (date - base_date).days
    return days_diff


def get_time_dataset():
    df = pd.read_excel("processed_data/time_sale_all.xlsx")
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    df['time'] = df['time'].apply(date_to_integer)
    time_volume = np.array(df).astype(float)
    cost_price = pd.read_excel("processed_data/cost_all.xlsx")
    cost_price = np.array(cost_price)[:, 1:].astype(float)
    cost = cost_price[:, range(0, 11, 2)]
    price = cost_price[:, range(1, 12, 2)]
    data = np.concatenate([time_volume, cost, price], axis=1)
    dataset = TimeDataSet(data)
    return dataset


def train_lstm_model(epochs=100, step=30, mode="part", save=False, save_loss=False):
    """
        train the lstm-ARIMA model
    """
    # model & loss & optim
    model = LSTMModel()
    model.fit(get_time_dataset())
    loss_func = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    filename = "pretrained_weight/lstm_weight_step_{}.pt".format(step)
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
   
    # dataset and epochs 
    if mode == "part":
        train_dataset = model.get_dataset("train", step)
        valid_dataset = model.get_dataset("valid", step)
        test_dataset = model.get_dataset("test", step)
        best_val_score = 10000
        test_loss_score = 0
    elif mode == "all":
        train_dataset = model.get_dataset("all", step)
    elif mode == "last":
        train_dataset = model.get_dataset("all", step)[-100:]
    
    model.train()
    train_epochs = trange(epochs, ascii=True, leave=True, desc="Epoch", position=0)
    
    loss_list = list()
    # train & valid
    for epoch in train_epochs:
        loss_score = 0
        pred_diff_list = list()
        y_diff_list = list()
        for data in train_dataset:
            x, x_corr, t, x_diff, _, y_diff = data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            optimizer.zero_grad()
            pred_diff = model(x, x_corr, t, x_diff)
            loss = loss_func(pred_diff, y_diff)
            pred_diff_list.append(pred_diff.cpu().detach().numpy())
            y_diff_list.append(y_diff.cpu().detach().numpy())
            loss.backward()
            loss_score += loss.item()
            optimizer.step()
        loss_score /= len(train_dataset)
        train_epochs.set_description("Epoch (Loss=%g)" % round(loss_score, 5))
        loss_list.append(loss_score)

        if mode == "all" and epoch == epochs-1:
            np_pred_diff = np.array(pred_diff_list)[:, 0][-100:]
            np_y_diff = np.array(y_diff_list)[:, 0][-100:]
            draw_arrays(np_pred_diff, np_y_diff, label1="pred", label2="fact",
                        save=True, filename="all_pred_lines_volume.png".format(epoch))
        # valid
        if mode == "part" and epoch % 10 == 9:
            val_loss_score = 0
            for data in valid_dataset:
                x, x_corr, t, x_diff, _, y_diff = data
                x = x.to(device)
                x_corr = x_corr.to(device)
                t = t.to(device)
                x_diff = x_diff.to(device)
                y_diff = y_diff.to(device)
                pred_diff = model(x, x_corr, t, x_diff)
                loss = loss_func(pred_diff, y_diff)
                val_loss_score += loss.item()
            val_loss_score /= len(valid_dataset)
            print("\nvalid loss score: ", val_loss_score)
            if val_loss_score < best_val_score:
                best_val_score = val_loss_score
                if save:
                    torch.save(model.state_dict(), filename)
                np_pred_diff = np.array(pred_diff_list)[:, 0][-100:]
                np_y_diff = np.array(y_diff_list)[:, 0][-100:]
                draw_arrays(np_pred_diff, np_y_diff, label1="pred", label2="fact",
                            save=True, filename="epoch_{}_pred_lines_volume.png".format(epoch))
    
    if save_loss:
        torch.save(torch.Tensor(loss_list), "pretrained_weight/train_loss.pt")
        draw_array(np.array(loss), save=True, filename="train_loss.png", 
            label="loss", x_label="Epoch", y_label="MSE loss", 
            title="lstm-ARIMA Model Train loss",
            noise_method=False)
        
    if save and (mode == "all" or mode == "last"):
        torch.save(model.state_dict(), filename)
    elif mode == "part":
        # test
        for data in test_dataset:
            x, x_corr, t, x_diff, _, y_diff = data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            pred_diff = model(x, x_corr, t, x_diff)
            loss = loss_func(pred_diff, y_diff)
            test_loss_score += loss.item()
        test_loss_score /= len(test_dataset)
        print("test loss score: ", test_loss_score)
        
        
def train_lstm_model_disease(epochs=100, step=30, mode="part", save=False, save_loss=False):
    """
        Train the LSTM-ARIMA Model Considering Disease
    """
    # model & loss & optim
    model = LSTMDiseaseModel()
    model.fit(get_time_dataset())
    loss_func = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    filename = "pretrained_weight/disease_lstm_weight_step_{}.pt".format(step)
    
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
    
    # dataset and epochs 
    if mode == "part":
        train_dataset = model.get_dataset("train", step)
        valid_dataset = model.get_dataset("valid", step)
        test_dataset = model.get_dataset("test", step)
        best_val_score = 10000
        test_loss_score = 0
    elif mode == "all":
        train_dataset = model.get_dataset("all", step)
    elif mode == "last":
        train_dataset = model.get_dataset("all", step)[-100:]
    
    model.train()
    train_epochs = trange(epochs, ascii=True, leave=True, desc="Epoch", position=0)
    
    loss_list = list()
    # train & valid
    for epoch in train_epochs:
        loss_score = 0
        pred_diff_list = list()
        y_diff_list = list()
        for data in train_dataset:
            x, x_corr, t, x_diff, _, y_diff, disease = data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            disease = disease.to(device)
            
            optimizer.zero_grad()
            pred_diff = model(x, x_corr, t, x_diff, disease)
            loss = loss_func(pred_diff, y_diff)
            pred_diff_list.append(pred_diff.cpu().detach().numpy())
            y_diff_list.append(y_diff.cpu().detach().numpy())
            loss.backward()
            loss_score += loss.item()
            optimizer.step()
            
        loss_score /= len(train_dataset)
        train_epochs.set_description("Epoch (Loss=%g)" % round(loss_score, 5))
        loss_list.append(loss_score)

        if mode == "all" and epoch == epochs-1:
            np_pred_diff = np.array(pred_diff_list)[:, 0][-100:]
            np_y_diff = np.array(y_diff_list)[:, 0][-100:]
            draw_arrays(np_pred_diff, np_y_diff, label1="pred", label2="fact",
                        save=True, filename="disease_all_pred_lines_volume.png".format(epoch))
        # valid
        if mode == "part" and epoch % 10 == 9:
            val_loss_score = 0
            for data in valid_dataset:
                x, x_corr, t, x_diff, _, y_diff, disease = data
                x = x.to(device)
                x_corr = x_corr.to(device)
                t = t.to(device)
                x_diff = x_diff.to(device)
                y_diff = y_diff.to(device)
                disease = disease.to(device)
                pred_diff = model(x, x_corr, t, x_diff, disease)
                loss = loss_func(pred_diff, y_diff)
                val_loss_score += loss.item()
            val_loss_score /= len(valid_dataset)
            print("\nvalid loss score: ", val_loss_score)
            if val_loss_score < best_val_score:
                best_val_score = val_loss_score
                if save:
                    torch.save(model.state_dict(), filename)
                np_pred_diff = np.array(pred_diff_list)[:, 0][-100:]
                np_y_diff = np.array(y_diff_list)[:, 0][-100:]
                draw_arrays(np_pred_diff, np_y_diff, label1="pred", label2="fact",
                            save=True, filename="disease_epoch_{}_pred_lines_volume.png".format(epoch))
    
    if save_loss:
        torch.save(torch.Tensor(loss_list), "pretrained_weight/disease_train_loss.pt")
        draw_array(np.array(loss), save=True, filename="disease_train_loss.png", 
            label="loss", x_label="Epoch", y_label="MSE loss", 
            title="lstm-ARIMA Disease Model Train loss",
            noise_method=False)
        
    if save and (mode == "all" or mode == "last"):
        torch.save(model.state_dict(), filename)
    elif mode == "part":
        # test
        for data in test_dataset:
            x, x_corr, t, x_diff, _, y_diff, disease = data
            x = x.to(device)
            x_corr = x_corr.to(device)
            t = t.to(device)
            x_diff = x_diff.to(device)
            y_diff = y_diff.to(device)
            disease = disease.to(device)
            pred_diff = model(x, x_corr, t, x_diff, disease)
            loss = loss_func(pred_diff, y_diff)
            test_loss_score += loss.item()
        test_loss_score /= len(test_dataset)
        print("test loss score: ", test_loss_score)