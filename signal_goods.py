import pandas as pd
import numpy as np
from arima import arima
from utils import get_category_id, get_category_name, get_goods


def get_single_predict():
    data = pd.read_excel("data/filter.xlsx").fillna(0)
    damage_info = pd.read_excel("data/data_4.xlsx")
    damage_info = np.array(damage_info)
    damage_info = damage_info[:, (0,2)]
    names = np.array(data.columns[1:])
    data = np.array(data)[:, 1:]
    ids = data[0].astype(np.int64)
    data = data[1:, :]
    predict_data = arima(data[-200:, :], predict_size=1)
    predict_data = np.expand_dims(predict_data[:, -1], axis=0).T 
    last_volume = np.expand_dims(data[-1, :], axis=0).T 
    
    last_price_list = list()
    last_cost_list = list()
    ela_list = list()
    pred_next_cost_list = list()
    damage_list = list()
    elasticity = np.load("processed_data/elasticity.npy", allow_pickle=True)
    trans_dict = {'辣椒类': 0,
                  '花叶类': 1,
                  '水生根茎类': 2,
                  '食用菌': 3,
                  '花菜类': 4,
                  '茄类': 5,}
    for id in ids:
        match_damage = damage_info[damage_info[:, 0] == id][0][1]
        damage_list.append(match_damage)
        ctg_name = get_category_name(id)
        ela_list.append(elasticity[trans_dict[ctg_name]])
        goods = get_goods(id)
        cost_info = goods.get_cost_info()
        cost_info = np.array(cost_info)[:, 1:]
        pred_next_cost_list.append(cost_info[-200:, 0])
        cost_info = cost_info[-1]
        if cost_info[2] == 0:
            new_cost_info = np.array(goods.get_cost_info())[:, 1:][:, 2]
            nonzero_indices = np.nonzero(new_cost_info)[0]
            last_nonzero_index = nonzero_indices[-1]
            last_price = new_cost_info[last_nonzero_index]
        else:
            last_price = cost_info[2]
        last_price_list.append(last_price)
        last_cost_list.append(cost_info[0])
        
    np_pred_next_cost = np.array(pred_next_cost_list).T
    next_cost = arima(np_pred_next_cost, predict_size=1)[:, -1]
    np_next_cost = np.expand_dims(next_cost, axis=0).T
    np_ela = np.expand_dims(np.array(ela_list), axis=0).T
    np_last_price = np.expand_dims(np.array(last_price_list), axis=0).T
    np_last_cost = np.expand_dims(np.array(last_cost_list), axis=0).T
    np_damage = np.expand_dims(np.array(damage_list), axis=0).T
    # caculate 
    np_data = np.concatenate([last_volume, predict_data, np_ela,\
        np_last_price, np_last_cost, np_next_cost, np_damage], axis=1)
    pd_data = pd.DataFrame(data=np_data, index=names, \
        columns=["前一天销量", "预测销量", "弹性系数", "前一天价格", \
            "前一天成本", "预测成本", "损耗率"])
    pd_data.to_excel("processed_data/单品数据.xlsx")
    


def _get_benifit(pred_volume, pred_cost, last_price, elasticity):
    def benifit_func(volume):
        diff_diff = volume - pred_volume
        alpha = diff_diff / pred_volume
        beta = alpha / elasticity
        price = (beta + 1) * last_price
        benifit = volume * (price - pred_cost)
        return benifit

    ub = pred_volume*1.05 + 1
    lb = pred_volume*0.95 - 1
    step_size = (ub - lb) / 100
    best_benifit = -1
    best_volume = pred_volume
    
    for i in range(100):
        volume = ub + step_size * i
        benifit = benifit_func(volume)
        if benifit > best_benifit:
            best_benifit = benifit
            best_volume = volume

    next_volume = best_volume
    next_diff_diff = next_volume - pred_volume
    next_alpha = next_diff_diff / pred_volume
    next_beta = next_alpha / elasticity
    next_price = next_beta * last_price + last_price
    benifit = next_volume * (next_price - pred_cost)
    return benifit, next_price, next_volume


def get_benifit():
    data = pd.read_excel("processed_data/单品数据.xlsx")
    data = np.array(data.iloc[:, 1:], dtype=float)
    pred_benifit_list = list()
    pred_price_list = list()
    define_volume_list = list()
    for i in range(data.shape[0]):
        cur_data = data[i]
        pred_benifit, pred_price, define_volume = _get_benifit(cur_data[1], \
            cur_data[5], cur_data[3], cur_data[2])
        pred_benifit_list.append(pred_benifit)
        pred_price_list.append(pred_price)
        define_volume_list.append(define_volume)
    
    np_pred_benifit = np.array(pred_benifit_list)
    np_pred_price = np.array(pred_price_list)
    np_define_volume = np.array(define_volume_list)
    
    df = pd.read_excel("processed_data/单品数据.xlsx")
    df["预测定价"] = np_pred_price
    df["修正预测销量"] = np_define_volume
    df["预测收益"] = np_pred_benifit
    df.to_excel("processed_data/预测收益.xlsx")
