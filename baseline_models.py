import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dataset import *
from utils import *
import wandb
import yaml
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def train_var_model(train_data, max_lag=1):
    model = VAR(train_data)
    results = model.fit(maxlags=max_lag)
    return results

def train_vecm_model(train_data, coint_rank=1, max_lag=1):
    model = VECM(train_data, k_ar_diff=max_lag, coint_rank=coint_rank)
    results = model.fit()
    
    return results

def forecast_var_model(model, last_observations, steps=1):
    forecast_data = model.forecast(last_observations, steps=steps)
    return forecast_data

def forecast_vecm_model(model, last_observations, steps=1):
    forecast_data = model.predict(steps=steps) #, alpha=last_observations
    return forecast_data

def test_models(test_data, lookback=30, max_lag=1):
    n_samples, n_features = test_data.shape
    n_test_samples = n_samples - lookback
    
    var_forecasts = []
    vecm_forecasts = []
    
    for i in range(n_test_samples):
        test_subset = test_data[i:i+lookback]
        last_observations = test_subset[-max_lag:]
        
        # Train VAR model
        var_model = train_var_model(test_subset, max_lag=max_lag)
        var_forecast = forecast_var_model(var_model, last_observations, steps=1)
        var_forecasts.append(var_forecast)
        
        # Train VECM model
        # vecm_model = train_vecm_model(test_subset, coint_rank=1, max_lag=max_lag)
        # vecm_forecast = forecast_vecm_model(vecm_model, last_observations, steps=1)
        # vecm_forecasts.append(vecm_forecast)
    
    var_forecasts = np.array(var_forecasts).reshape(n_test_samples, n_features)
    # vecm_forecasts = np.array(vecm_forecasts).reshape(n_test_samples, n_features)
    
    return var_forecasts #, vecm_forecasts

if __name__ == '__main__':
    # Assuming you have your test data in the variable 'test_data' with shape (6030, 11)
    # Load config file
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    

    ## ----------------------------------------------------------------------------------------------------------------------------
    # Log to wandb
    if cfg['log_wandb']:
        wandb.login()
        wandb.init(project=cfg['project'],
                name=cfg['run_name'],
                entity=cfg['maintainer'],
                config=cfg)
    
    ## ----------------------------------------------------------------------------------------------------------------------------    
    # Identify GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    print(f"Hello {cfg['name']} using {device}")
    
    ## ----------------------------------------------------------------------------------------------------------------------------    
    # Load Data
    # (train_data, train_dates, val_data, val_dates, test_data, test_dates), scaler = load_data(filepath = cfg['dataset']['filepath'],
    #                                                            test_size = cfg['dataset']['test_size'], interpolation=cfg['dataset']['interpolation'], interpolation_axis=cfg['dataset']['interpolation_axis'])
    
    (train_data, train_dates, val_data, val_dates, test_data, test_dates), scaler = load_data(filepath = cfg['dataset']['filepath'],
                                                               test_size = cfg['dataset']['test_size'])
    
    print(train_data.shape, val_data.shape, test_data.shape)
    
     ## ----------------------------------------------------------------------------------------------------------------------------
    # Train Test Split
    ##(lookback window, time steps, features))
    X_train, y_train = create_dataset(train_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    X_val, y_val = create_dataset(val_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    X_test, y_test = create_dataset(test_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    
    print(f'Train Data {X_train.shape, y_train.shape} Val Data {X_val.shape, y_val.shape} Test Data {X_test.shape, y_test.shape}')
    
    
    
    # test_data = ...  # Replace with your actual test data
    
    lookback = 30
    # var_forecasts, vecm_forecasts = test_models(test_data, lookback)
    var_forecasts = test_models(test_data, lookback)
    
    
    
    print(f"VAR forecasts shape: {var_forecasts.shape}")
    # print(f"VECM forecasts shape: {vecm_forecasts.shape}")
    
    
    
    y_true = y_test
    y_pred = var_forecasts
    print(f'Y pred {y_pred.shape, type(y_pred)}, Y true {y_true.shape, type(y_true)}')
    
    n_features = 11
    # y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, n_features))
    # y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, n_features))
    
    y_pred = np.array(y_pred).reshape(-1, n_features)
    y_true = np.array(y_true).reshape(-1, n_features)
    
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, n_features))
    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, n_features))
    test_iv_df = pd.concat([test_dates[cfg['dataset']['n_lookback']:].reset_index(drop=True), pd.DataFrame(np.array(y_true).reshape(-1, n_features)).reset_index(drop=True)], axis=1)
    pred_iv_df = pd.concat([test_dates[cfg['dataset']['n_lookback']:].reset_index(drop=True), pd.DataFrame(np.array(y_pred).reshape(-1, n_features)).reset_index(drop=True)], axis=1)
    
    plot_results(test_iv_df, pred_iv_df, model='VAR', save_path='outputs/figures/' + 'vecm_4h' + '.png', noise = 2)
    plot_surface(test_iv_df, pred_iv_df, model='VAR', n=1, save_path='outputs/figures/' + 'vecm_4h' + '.png', noise=2)
    
    # print(y_pred.shape, y_true.shape)
    
    mse = mean_squared_error(np.array(y_true).reshape(-1, n_features), np.array(y_pred).reshape(-1, n_features))
    # mse = mean_squared_error(y_true, y_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mda, mda_dict = mean_directional_accuracy(y_true, y_pred)
    
    print('MAE', mae)
    print('MDA', mda)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAPE: ', mape)
    print('R2: ', r2)
    print('MDA Dict: ', mda_dict)
    
    print(f'Y true: {y_true.shape} Y pred {y_pred.shape}')
    
    # wandb.log({'Test MAE': mae})
    # wandb.log({'Test MDA': mda})
    # wandb.log({'Test MSE': mse})
    # wandb.log({'Test MAPE': mape})
    # wandb.log({'Test RMSE': rmse})
    # wandb.log({'Test R2': r2})
    # wandb.log(mda_dict)
    
    # plt.plot(y_pred[:, 5][:100])
    # plt.plot(y_true[:, 5][:100])
    # plt.legend(['Y pred', 'Y True'])
    # plt.savefig('test.png')