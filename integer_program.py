import pulp
import numpy as np
import pandas as pd

def lip():
    """
    Linear Integer Programming
    """
    # load data
    df = pd.read_excel("processed_data/预测收益.xlsx")
    data = np.array(df)
    damage = data[:, -4]
    benifits = data[:, -1]
    volumes = data[:, -2]
    prices = data[:, -3]
    min_zero = np.empty(shape=(34,))
    for i in range(34):
        min_zero[i] = min(volumes[i]-2.5, 0) 
    fix_benifits = benifits +  min_zero * prices
    fix_benifits -= (damage/100) * volumes * prices * 0.1
    # create problem
    problem = pulp.LpProblem("Goods Selected", pulp.LpMaximize)
    # target
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpInteger) for i in range(34)]
    # object
    object = pulp.lpSum(x[i] * fix_benifits[i] for i in range(34))
    problem += object
    # constraint
    problem += pulp.lpSum(x[i] for i in range(34)) <= 33
    problem += pulp.lpSum(x[i] for i in range(34)) >= 27
    # get result
    problem.solve()
    result = np.zeros(shape=(34, ))
    for i in range(34):
        result[i] = pulp.value(x[i])
        
    df = pd.read_excel("processed_data/预测收益.xlsx")
    df["修正收益"] = fix_benifits
    df["是否选择"] = result
    df.to_excel("processed_data/选择结果.xlsx")
    
    return result    

