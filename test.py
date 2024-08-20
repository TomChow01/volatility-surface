from tqdm import tqdm
import torch
import torch.nn as nn

import numpy as np
import wandb



def test(cfg, model, test_loader, device):
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
            y_batch = y_batch.reshape(-1, cfg['model']['n_features'])
            # print(y_batch)
            # break
            y_pred = model(X_batch)
     
            
            # print('GT %.2f Pred %.2f' %(y_batch, y_pred))
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item()
            
            y_pred = y_pred.detach().cpu()
            y_batch = y_batch.detach().cpu()
            # print(y_pred, y_batch)
            
            y_preds.append(y_pred)
            y_true.append(y_batch)
        
        test_loss /= len(test_loader)
        wandb.log({'Test Loss': np.sqrt(test_loss)})
        print(f'Test Loss: {test_loss:.4f}')
        return y_true, y_preds