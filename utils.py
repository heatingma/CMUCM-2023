import numpy as np
import pandas as pd


###################################################
#                   data process                  #
###################################################

try:
    category = np.load("processed_data/category.npy", allow_pickle=True).item()
except FileNotFoundError:
    category = dict()
    data = np.array(pd.read_excel("data/data_1.xlsx"))
    for item in data:
        category[item[0]] = (item[1], item[2], item[3])
    np.save("processed_data/category.npy", category)

try:
    sales_info = np.load("processed_data/sales_info.npy", allow_pickle=True)
except FileNotFoundError:
    sales_info = np.array(pd.read_excel("data/data_2.xlsx"))
    np.save("processed_data/sales_info.npy", sales_info)


###################################################
#        get item & category info function        #
###################################################


def get_goods_name(item_id):
    return category[item_id][0]


def get_category_id(item_id):
    return category[item_id][1]


def get_category_name(item_id):
    return category[item_id][2]


def get_categorys():
    categorys = CATEGORYS()
    for sale_info in sales_info:
        categorys.add_sale_info(sale_info[0], sale_info[2], sale_info[3], \
            sale_info[4], sale_info[5], sale_info[6])
    return categorys


###################################################
#                    utils class                  #
###################################################

class GOODS:
    def __init__(self, id):
        self.id = id
        self.sales_info = list()
    
    def add_sale_info(self, date, volume, unit_price, sale_type, discount):
        self.sales_info.append([date, volume, unit_price, sale_type, discount])

    def get_time_sale_info(self):
        sales_info = np.array(self.sales_info)
        sales_info = pd.DataFrame(data=sales_info[:, 1], index=sales_info[:, 0], \
            columns=[get_goods_name(self.id)])
        
        sales_info.index = pd.to_datetime(sales_info.index)
        date_range = pd.date_range(start="2020-07-01", end="2023-06-30", freq="D")
        sales_info = sales_info.groupby(sales_info.index).sum()
        sales_info = sales_info.reindex(date_range, fill_value=0)
        
        return sales_info
            
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
        all_sales_info.to_excel("processed_data/time_sale_all.xlsx")
    
    def __repr__(self):
        message = "category_dict"
        return f"{self.__class__.__name__}({message})"  
    