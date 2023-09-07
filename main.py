import pandas as pd
import numpy as np
from utils import get_categorys
from models import arima


def problem_1():
    data = get_categorys()
    data.get_time_sale_info()


def problem_2():
    time_sale_all = pd.read_excel("processed_data/time_sale_all.xlsx")
    time_sale_all = np.array(time_sale_all)
    for i in range(6):
        arima(time_sale_all[:, i+1])
    
    
if __name__ == "__main__":
    # problem_1()
    problem_2()