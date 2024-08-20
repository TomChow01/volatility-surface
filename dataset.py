import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch


def load_data(filepath, test_size=0.3):
    iv_data = pd.read_csv(filepath)
    data = iv_data[['Datetime', 'Expiry', 'Price', 'IV_ITM_5', 'IV_ITM_4', 'IV_ITM_3', 'IV_ITM_2', 'IV_ITM_1', 'IV_ATM', 'IV_OTM_1', 'IV_OTM_2', 'IV_OTM_3', 'IV_OTM_4', 'IV_OTM_5']]
    data = data.replace(0, np.nan)
    
    data = data.interpolate(method='linear')
    data.dropna(inplace=True)
    dates = data[['Datetime', 'Expiry']]
    data = data[['IV_ITM_5', 'IV_ITM_4', 'IV_ITM_3', 'IV_ITM_2', 'IV_ITM_1', 'IV_ATM', 'IV_OTM_1', 'IV_OTM_2', 'IV_OTM_3', 'IV_OTM_4', 'IV_OTM_5']]
    data = data.values.astype('float32')

    
    
    # train-test split for time series
    train_size = int(len(data) * (1-test_size))
    test_size = len(data) - train_size

    # train_X, train_y = data[:train_size], labels[:train_size]
    # test_X, test_y = data[train_size:], labels[train_size:]

    train_data, train_dates = data[:train_size], dates[:train_size]  #, labels[:train_size]
    val_data, val_dates = data[train_size:train_size+int(0.33*test_size)], dates[train_size:train_size+int(0.33*test_size)]
    test_data, test_dates = data[train_size+int(0.33*test_size):], dates[train_size+int(0.33*test_size):] #, labels[train_size:]
    
    
    scaler = MinMaxScaler(feature_range=(0, 1)) # RobustScaler() 
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # print(train_data.shape, train_dates.shape) 
    # print(test_data.shape, test_dates.shape)

    return (train_data, train_dates, val_data, val_dates, test_data, test_dates), scaler


def create_dataset(dataset, n_lookback, n_forecast):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        n_lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-n_lookback): 
        feature = dataset[i:i+n_lookback]
        # target = dataset[i+1:i+n_lookback+1]
        target = dataset[i+n_lookback:i+n_lookback+n_forecast]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)