import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

def get_arima_d(data: pd.DataFrame):
    d = 0
    while True:
        adf_result = adfuller(data["diff_{}".format(d)].dropna())
        p = adf_result[1]
        if p < 0.05:
            return d
        if d >= 2:
            raise ValueError("too many diffs")
        d += 1  
        data["diff_{}".format(d)] = data["diff_{}".format(d-1)].diff(1)
        
        
def get_arima_p_q(data: pd.DataFrame, arima_d):
    pmax = 2
    qmax = 2
    bic_matrix  =  []
    np_data = np.nan_to_num(data["diff_0"], nan=0).astype(float)
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(np_data, order=(p, arima_d, q)).fit().bic) 
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().idxmin()  
    return p, q


def _arima(data: np.ndarray, predict_size=0):
    """
        Predict the future data using ARIMA Modle
    """
    # data process
    df = pd.DataFrame(data=data, columns=["diff_0"])
    # define the d
    d = get_arima_d(df)
    # define the p & q
    p, q = get_arima_p_q(df, d)
    # arima model
    model = ARIMA(data.astype(float), order=(p, d, q)).fit()
    # predict 
    fittedvalues = model.fittedvalues
    predict = model.forecast(predict_size+1)
    result = np.concatenate([np.expand_dims(fittedvalues, axis=0), 
                   np.expand_dims(predict, axis=0)], axis=1).squeeze(axis=0)
    result = result[1:]
    return result


def arima(data: np.ndarray, predict_size=0):
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    arima_pred = list()
    for i in range(data.shape[1]):
        arima_pred.append(_arima(data[:, i], predict_size))
    arima_pred = np.array(arima_pred)
    return arima_pred
