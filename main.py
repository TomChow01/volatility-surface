import yaml
import wandb
import random
import plotly.graph_objs as go

# importing libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



from utils import *
from dataset import *
from model import *
from train import *
from test import *

import torch
from torch.utils.data import DataLoader, TensorDataset




def main(config_file: str = None):
    
    # Load config file
    with open(config_file) as f:
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
    
    ## ----------------------------------------------------------------------------------------------------------------------------
    # Train Test Split
    ##(lookback window, time steps, features))
    X_train, y_train = create_dataset(train_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    X_val, y_val = create_dataset(val_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    X_test, y_test = create_dataset(test_data, n_lookback=cfg['dataset']['n_lookback'], n_forecast=cfg['dataset']['n_forecast'])
    
    print(f'Train Data {X_train.shape, y_train.shape} Val Data {X_val.shape, y_val.shape} Test Data {X_test.shape, y_test.shape}')
    
    
    ## ----------------------------------------------------------------------------------------------------------------------------
    # Model
    n_features = X_train.shape[-1]
    # Define the model
    if cfg['model']['type'] == 'simple_lstm':
        model = Model(cfg)
    elif cfg['model']['type'] == 'att_lstm':
        model = AttLSTM(cfg) #Model(cfg)


    ## ----------------------------------------------------------------------------------------------------------------------------
    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=cfg['training']['batch_size'])
    val_loader = DataLoader(TensorDataset(X_val, y_val), shuffle=True, batch_size=cfg['training']['batch_size'])
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=1)

    
    ## ----------------------------------------------------------------------------------------------------------------------------
    # Training
    if cfg['train']:
        trainer = Trainer(cfg, model, train_loader, val_loader, device)
        trainer.train()
    
    ## Testing
    if cfg['test']:
        # trainer.test_model(test_loader)
        if cfg['model']['type'] == 'simple_lstm':
            model = Model(cfg)
        elif cfg['model']['type'] == 'att_lstm':
            model = AttLSTM(cfg) #Model(cfg)
        # model = AttLSTM(cfg) #Model(cfg) # AttLSTM(cfg) #
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(cfg['model']['save_path'], 'best_model.pt')))
        y_true, y_pred = test(cfg, model, test_loader, device, scaler=None)
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, n_features))
        y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, n_features))
        test_iv_df = pd.concat([test_dates[cfg['dataset']['n_lookback']:].reset_index(drop=True), pd.DataFrame(np.array(y_true).reshape(-1, n_features)).reset_index(drop=True)], axis=1)
        pred_iv_df = pd.concat([test_dates[cfg['dataset']['n_lookback']:].reset_index(drop=True), pd.DataFrame(np.array(y_pred).reshape(-1, n_features)).reset_index(drop=True)], axis=1)
        plot_results(test_iv_df, pred_iv_df)
        
    
    plot_surface(test_iv_df, pred_iv_df, n = 3)
    
    wandb.finish()
    
    
if __name__ == '__main__':
    ## TODO: MSE, MAE, MAPE
    ## TODO: ARCH, GARCH and Auto-Regression
    ## TODO: EDA
    main('config.yaml')