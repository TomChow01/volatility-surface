from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *

import numpy as np
import wandb

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score



def test(cfg, model, test_loader, device, scaler=None):
    print("Testing model...")
    if cfg['training']['loss'] == 'mse':   
        loss_fn = nn.MSELoss()
            
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        y_true = []
        y_preds = []
        for X_batch, y_batch in tqdm(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.reshape(-1, cfg['dataset']['n_forecast'], cfg['model']['n_features'])
            # print(X_batch.shape, y_batch.shape)
            # break
            y_pred = model(X_batch)
            # y_pred = y_pred.reshape(-1, cfg['model']['n_features'])
            # print(y_pred.shape, y_batch.shape)
     
            
            # print('GT %.2f Pred %.2f' %(y_batch, y_pred))
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item()
            
            y_pred = y_pred.detach().cpu().numpy()#[0]
            y_batch = y_batch.detach().cpu().numpy()#[0]
            # print(y_pred.shape, y_batch.shape)
            # break
            
            y_preds.append(y_pred)
            y_true.append(y_batch)
        
        # mda = mean_directional_accuracy(y_true, y_preds)
        # mae = mean_absolute_error(y_true, y_preds)
    
        # test_loss /= len(test_loader)
        # print('y_preds',y_preds[:10])
        y_preds = np.concatenate(y_preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # y_preds = y_preds[:-39]
        # y_true = y_true[:-39]
        
        # print('len:', len(y_preds), np.shape(y_preds[-1]))
        # y_preds = np.array(y_preds)
        # y_true = np.array(y_true)
        
        # y_preds = y_preds.reshape(-1, y_preds.shape[2], y_preds.shape[3])
        # y_true = y_true.reshape(-1, y_true.shape[2], y_true.shape[3])
        
        # print('Before: ', y_true.shape, y_preds.shape)
        
        
        y_preds = y_preds[:, -1, :]
        y_true = y_true[:, -1, :]
        
        if scaler:
            y_preds = scaler.inverse_transform(np.array(y_preds))
            y_true = scaler.inverse_transform(np.array(y_true))
        
        # print(y_true.shape, y_preds.shape)
        
            
        mse = mean_squared_error(y_true, y_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_preds)
        mape = mean_absolute_percentage_error(y_true, y_preds)
        mda, mda_dict = mean_directional_accuracy(y_true, y_preds)
        r2 = r2_score(y_true, y_preds)
        
        df = pd.DataFrame({'rmse': [rmse], 'mape': [mape], 'mae': [mae], 'r2': [r2], 'mda': [mda]})
        df.to_csv(cfg['test']['result_path'] + cfg['run_name'] + '.csv')
        
        print('MAE', mae)
        print('MDA', mda)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('MAPE: ', mape)
        print('R2: ', r2)
        print('mda dict: ', mda_dict)
        
        
        wandb.log({'Test MAE': mae})
        wandb.log({'Test MDA': mda})
        wandb.log({'Test MSE': mse})
        wandb.log({'Test MAPE': mape})
        wandb.log({'Test RMSE': rmse})
        wandb.log({'Test R2': r2})
        wandb.log(mda_dict)
        # print(f'Test Loss: {test_loss:.4f}')
        return y_true, y_preds