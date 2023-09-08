from utils import get_categorys
from models import NARXModel
from get_dataset import get_dataset
import torch.nn as nn
import torch
from tqdm import trange
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_info():
    data = get_categorys()
    data.get_time_sale_info()
    data.get_cost_info()
    

def train_narx_model(epochs=100, step=30, mode="part"):
    """
        train the NARX-ARIMA model
    """
    # model & loss & optim
    model = NARXModel()
    loss_func = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    filename = "pretrained_weight/narx_weight_step_{}.pt".format(step)
    model.load_state_dict(torch.load(filename))
   
    # dataset and epochs 
    if mode == "part":
        train_dataset = get_dataset("train", step)
        valid_dataset = get_dataset("valid", step)
        test_dataset = get_dataset("test", step)
        best_val_score = 10000
        test_loss_score = 0
    elif mode == "all":
        train_dataset = get_dataset("all", step)
    
    model.train()
    train_epochs = trange(epochs, ascii=True, leave=True, desc="Epoch", position=0)
   
    # train & valid
    for epoch in train_epochs:
        loss_score = 0
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
            loss.backward()
            loss_score += loss.item()
            optimizer.step()
        loss_score /= len(train_dataset)
        train_epochs.set_description("Epoch (Loss=%g)" % round(loss_score, 5))
        
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
                torch.save(model.state_dict(), filename)
    
    if mode == "all":
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
    
 
if __name__ == "__main__":
    get_data_info()
    train_narx_model(mode="part")
    train_narx_model(mode="all")