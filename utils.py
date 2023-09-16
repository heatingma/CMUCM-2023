import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

###################################################
#                   data process                  #
###################################################

try:
    category = np.load("processed_data/category.npy", allow_pickle=True).item()
except FileNotFoundError:
    category = dict()
    data = np.array(pd.read_excel("data/data_1.xlsx").dropna())
    for item in data:
        category[item[0]] = (item[1], item[2], item[3])
    np.save("processed_data/category.npy", category)

try:
    sales_info = np.load("processed_data/sales_info.npy", allow_pickle=True)
except FileNotFoundError:
    sales_info = np.array(pd.read_excel("data/data_2.xlsx").dropna())
    np.save("processed_data/sales_info.npy", sales_info)

try:
    cost_info = np.load("processed_data/cost_info.npy", allow_pickle=True)
except FileNotFoundError:
    cost_info = np.array(pd.read_excel("data/data_3.xlsx").dropna())
    np.save("processed_data/cost_info.npy", cost_info)

    
###################################################
#        get item & category info function        #
###################################################

def get_goods(item_id):
    categorys = get_categorys()
    ctg_id = get_category_id(item_id)
    ctg = categorys.category_dict[ctg_id]
    ctg: CATEGORY
    goods = ctg.goods_dict[item_id]
    goods: GOODS
    return goods


def get_goods_name(item_id):
    return category[item_id][0]


def get_category_id(item_id):
    return category[item_id][1]


def get_category_name(item_id):
    return category[item_id][2]


def category_name2id(name:str):
    trans_dict = {'辣椒类':1011010504, 
                  '花叶类':1011010101,
                  '花菜类':1011010201,
                  '水生根茎类':1011010402,
                  '食用菌':1011010801,
                  '茄类':1011010501}
    return trans_dict[name]


def get_categorys():
    categorys = CATEGORYS()
    for sale_info in sales_info:
        categorys.add_sale_info(sale_info[0], sale_info[2], sale_info[3], \
            sale_info[4], sale_info[5], sale_info[6])
    for part_cost_info in cost_info:
        categorys.add_cost(part_cost_info[0], part_cost_info[1], part_cost_info[2])
    return categorys


###################################################
#                    utils class                  #
###################################################

class GOODS:
    def __init__(self, id):
        self.id = id
        self.sales_info = list()
        self.cost_info = list()
        
    def add_sale_info(self, date, volume, unit_price, sale_type, discount):
        self.sales_info.append([date, volume, unit_price, sale_type, discount])

    def add_cost(self, date, cost):
        self.cost_info.append([date, cost])
        
    def get_time_sale_info(self):
        sales_info = np.array(self.sales_info)
        sales_info = pd.DataFrame(data=sales_info[:, 1], index=sales_info[:, 0], \
            columns=[get_goods_name(self.id)])
        sales_info.index = pd.to_datetime(sales_info.index)
        date_range = pd.date_range(start="2020-07-01", end="2023-06-30", freq="D")
        sales_info = sales_info.groupby(sales_info.index).sum()
        sales_info = sales_info.reindex(date_range)
        self.df_time_sale_info = sales_info
        return sales_info

    def get_price_info_info(self):
        sales_info = np.array(self.sales_info)
        data = np.array(sales_info[:, 2] * sales_info[:, 1], dtype=float)
        price_info = pd.DataFrame(data=data, index=sales_info[:, 0], \
            columns=[get_goods_name(self.id)])
        price_info.index = pd.to_datetime(price_info.index)
        date_range = pd.date_range(start="2020-07-01", end="2023-06-30", freq="D")
        price_info = price_info.groupby(price_info.index).sum()
        price_info = price_info.reindex(date_range)
        return price_info
    
    def get_cost_info(self):
        date_range = pd.date_range(start="2020-07-01", end="2023-06-30", freq="D")
        cost_info = pd.DataFrame(self.cost_info, columns=['time', 'cost'])
        cost_info['time'] = pd.to_datetime(cost_info['time'])
        cost_info = cost_info.set_index('time').reindex(date_range).reset_index()
        cost_info['cost'] = cost_info['cost'].fillna(method='ffill')
        cost_info['cost'] = cost_info['cost'].fillna(method='bfill')
        volume = np.array(self.get_time_sale_info().iloc[:, 0])
        prices = np.array(self.get_price_info_info().iloc[:, 0])
        cost_info['volume'] = volume
        cost_info['price'] = np.nan_to_num(prices / volume)
        cost_info = cost_info.rename(columns={cost_info.columns[0]: 'time'}).fillna(0)
        return cost_info
    
    def __repr__(self):
        message = "id, sales_info"
        return f"{self.__class__.__name__}({message})"  
        

class CATEGORY:
    def __init__(self, id) -> None:
        self.id = id
        self.id_name = None
        self.goods_dict = dict()
        
    def add_sale_info(self, date, goods_id, volume, \
        unit_price, sale_type, discount):
        if goods_id not in self.goods_dict.keys():
            self.goods_dict[goods_id] = GOODS(goods_id)
        self.goods_dict[goods_id].add_sale_info(date, volume, \
            unit_price, sale_type, discount)
        if self.id_name is None:
            self.id_name = get_category_name(goods_id)
    
    def add_cost(self, date, goods_id, cost):
        if goods_id not in self.goods_dict.keys():
            return
        self.goods_dict[goods_id].add_cost(date, cost)
            
    def get_time_sale_info(self):
        ctg_sales_info = None
        for goods in self.goods_dict.values():
            goods: GOODS
            goods_sales_info = goods.get_time_sale_info()
            if ctg_sales_info is None:
                ctg_sales_info = goods_sales_info
            else:
                ctg_sales_info = pd.concat([ctg_sales_info, goods_sales_info], axis=1)
        ctg_sales_info[self.id_name] = ctg_sales_info.sum(axis=1)
        return ctg_sales_info

    def get_cost_info(self):
        index = None
        sum_cost = None
        sum_volume = None
        sum_price = None
        for goods in self.goods_dict.values():
            goods: GOODS
            goods_cost_info = goods.get_cost_info()
            if index is None:
                index = np.array(goods_cost_info['time'])
                sum_volume = np.array(goods_cost_info['volume'])
                sum_cost = np.array(goods_cost_info['cost']) * sum_volume
                sum_price = np.array(goods_cost_info['price']) * sum_volume
            else:
                cur_volume = np.array(goods_cost_info['volume'])
                sum_volume += cur_volume
                sum_cost += cur_volume * np.array(goods_cost_info['cost'])
                sum_price += cur_volume * np.array(goods_cost_info['price'])

        data = np.concatenate([sum_cost/sum_volume, \
            sum_price/sum_volume], axis=0).reshape(2, -1).T
        ctg_cost_info = pd.DataFrame(data=data, index=index, \
            columns=[self.id_name+"平均成本", self.id_name+"平均售价"])
        return ctg_cost_info
    
    def __repr__(self):
        message = "id, goods_dict"
        return f"{self.__class__.__name__}({message})"  
    

class CATEGORYS:
    def __init__(self) -> None:
        self.category_dict = dict()
        
    def add_sale_info(self, date, goods_id, volume, \
        unit_price, sale_type, discount):
        category_id = get_category_id(goods_id)
        if category_id not in self.category_dict.keys():
            self.category_dict[category_id] = CATEGORY(category_id)
        self.category_dict[category_id].add_sale_info(date, goods_id, volume, \
        unit_price, sale_type, discount)
    
    def add_cost(self, date, goods_id, cost):
        category_id = get_category_id(goods_id)
        if category_id not in self.category_dict.keys():
            return
        self.category_dict[category_id].add_cost(date, goods_id, cost)
        
    def get_time_sale_info(self):
        all_sales_info = None
        for ctg in self.category_dict.values():
            ctg: CATEGORY
            ctg_sales_info = ctg.get_time_sale_info()
            ctg_sales_info.to_excel("processed_data/time_sale_{}.xlsx".format(ctg.id_name))
            if all_sales_info is None:
                all_sales_info = ctg_sales_info[ctg.id_name]
            else:
                all_sales_info = pd.concat([all_sales_info, ctg_sales_info[ctg.id_name]], axis=1)
        all_sales_info.dropna(inplace=True)
        all_sales_info = all_sales_info[~(all_sales_info == 0).any(axis=1)]
        all_sales_info.to_excel("processed_data/time_sale_all.xlsx")

    def get_cost_info(self):
        all_cost_info = None
        for ctg in self.category_dict.values():
            ctg: CATEGORY
            ctg_cost_info = ctg.get_cost_info()
            if all_cost_info is None:
                all_cost_info = ctg_cost_info
            else:
                all_cost_info = pd.concat([all_cost_info, ctg_cost_info], axis=1)
        all_cost_info.dropna(inplace=True)
        all_cost_info.to_excel("processed_data/cost_all.xlsx")
        return all_cost_info
       
    def __repr__(self):
        message = "category_dict"
        return f"{self.__class__.__name__}({message})"  


###################################################
#                   draw picture                  #
###################################################

def draw_xy(x: np.ndarray, y:np.ndarray, save=False, filename="draw_xy.png"):
    point_size = 2
    plt.scatter(x, y, s=point_size)
    plt.title("Scatter Plot of x and y")
    plt.xlabel("x")
    plt.ylabel("y")
    if save:
        plt.savefig(os.path.join("pics", filename))
    plt.clf()
        

     
def draw_arrays(array: np.ndarray, array2: np.ndarray, 
                label1='Array 1', label2='Array 2',
                save=False, filename="draw_arrays.png"):
    x = np.arange(len(array))
    plt.figure(figsize=(30, 10))
    plt.plot(x, array, marker='o', label=label1)
    plt.plot(x, array2, marker='o', label=label2)
    plt.title('One-dimensional Vector')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    if save:
        plt.savefig(os.path.join("pics", filename))
    plt.clf()

def draw_three_arrays(array: np.ndarray, array2: np.ndarray, array3: np.ndarray,
                label1='Array 1', label2='Array 2', label3='Array 3',
                save=False, filename="draw_arrays.png"):
    plt.rcParams.update({'font.size': 16})
    x = np.arange(len(array))
    plt.figure(figsize=(30, 10))
    plt.plot(x, array, marker='o', label=label1)
    plt.plot(x, array2, marker='o', label=label2)
    plt.plot(x, array3, marker='o', label=label3)
    plt.title('Predictions compare')
    plt.xlabel('time')
    plt.ylabel('predict')
    plt.legend()
    if save:
        plt.savefig(os.path.join("pics", filename))
    plt.clf()

def draw_array(array: np.ndarray, save=False, filename="draw_array.png", figsize=(15, 8), 
               label="array", x_label='Index', y_label='Value', title="One-dimensional Vector", noise_method=True):
    plt.figure(figsize=figsize)
    if noise_method:
        mean_array = abs(np.average(array))
        array = array[~(abs(array) >= mean_array * 20)]
    x = np.arange(len(array))
    plt.plot(x, array, marker='o', markersize=3, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig(os.path.join("pics", filename))
    plt.clf()
