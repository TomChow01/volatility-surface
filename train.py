import matplotlib.pyplot as plt
from model import *
# from utils import mape
from torchmetrics.regression import MeanAbsolutePercentageError
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import wandb
import os

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score




class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        if self.cfg['training']['optimizer'] == 'Adam':   
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.cfg['training']['lr'])
        if self.cfg['training']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.cfg['training']['lr'], momentum=0.9)
        
        if self.cfg['training']['loss'] == 'mse':   
            self.loss_fn = nn.MSELoss()
        if self.cfg['training']['loss'] == 'mae':
            self.loss_fn = nn.L1Loss()
        
        # self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.mape = MeanAbsolutePercentageError()
            
        ## Scheduler
        if cfg['training']['scheduler'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        elif cfg['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif cfg['training']['scheduler'] == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer)
        elif cfg['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10) 
        elif cfg['training']['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=8)
        else:
            print('No scheduler specified')  
            
        
        # Save Path
        self.save_path = self.cfg['model']['save_path']     
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)                                                                                      
            
        self.model.to(self.device)
    
    
    def train(self):


        train_losses, val_losses, test_losses = [], [], []
        min_val_loss = np.inf
        
        #TODO: In-sample Prediction
        #TODO: Early Stopping
        for epoch in range(1, self.cfg['training']['n_epochs']+1):
            self.model.train()
            epoch_mse = 0
            epoch_mae = 0
            
            ## Batchwise Training
            for X_batch, y_batch in tqdm(self.train_loader):
                X_batch = X_batch.to(self.device)
                y_batch =  y_batch.to(self.device).squeeze(1) # y_batch.reshape(-1, 1).to(device)
                # print(X_batch.shape, y_batch.shape)
                
                y_pred = self.model(X_batch)
                # print(y_pred.shape, y_batch.shape)
                loss = self.loss_fn(y_pred, y_batch)
                mae_loss = self.mae(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f'{name} grad norm: {param.grad.norm().item()}')
                        
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient Clipping
                # or
                if self.cfg['training']['clip_gradient']:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg['training']['clip_gradient_value'])
                
                self.optimizer.step()
                epoch_mse += loss.item()
                epoch_mae += mae_loss.item()
            
            self.scheduler.step()
            wandb.log({'Learning Rate': self.optimizer.param_groups[0]['lr']})
                
            epoch_rmse = np.sqrt(epoch_mse) / len(self.train_loader) #RMSE
            epoch_mae = epoch_mae /len(self.train_loader)
            train_losses.append(epoch_rmse)
            wandb.log({'Train Loss': epoch_rmse}) #/len(self.train_loader)})
            wandb.log({'Train Loss (MAE)': epoch_mae})
            print(f'Epoch {epoch} Loss (RMSE | MAE) {epoch_rmse, epoch_mae}')
            
            # Validation every 10 epochs
            # if epoch % 10 == 0:
                # continue
                
              
            self.model.eval()
            val_mse, val_mae, val_mape = 0, 0, 0
            val_gts, val_preds = [], []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch =  y_batch.to(self.device).squeeze(1)                                                                                    
                    y_pred = self.model(X_batch.to(self.device)) #.detach().cpu()
                    
                    val_gts.append(y_batch.detach().cpu().numpy())
                    val_preds.append(y_pred.detach().cpu().numpy())
                    
                    val_mse += self.loss_fn(y_pred, y_batch).item()
                    val_mae += self.mae(y_pred, y_batch).item()
                    val_mape += self.mape(y_pred.detach().cpu(), y_batch.detach().cpu()).item()
                    
                    # test_loss = 0 #np.sqrt(loss_fn(y_pred, y_test.reshape(-1, 1)))
            
            
            val_gts, val_preds = np.concatenate(val_gts), np.concatenate(val_preds)
            print('SKLEARN')
            print(mean_squared_error(val_gts, val_preds), mean_absolute_error(val_gts, val_preds), mean_absolute_percentage_error(val_gts, val_preds), r2_score(val_gts, val_preds))
            # print(y_pred.shape, y_batch.shape)
            plt.figure(figsize=(10, 5))
            plt.plot(y_pred.detach().cpu().numpy()[16])
            plt.plot(y_batch.detach().cpu().numpy()[16])
            plt.legend(['y_pred', 'y_true'])
            plt.title('Epoch %d: Predictions vs. True Values' % epoch)
            wandb.log({'Predictions': wandb.Image(plt)})
            val_rmse = np.sqrt(val_mse)
            val_losses.append(val_rmse) #/len(self.train_loader))
            wandb.log({'Val MSE': val_mse/len(self.val_loader), 'Val RMSE': val_rmse/len(self.val_loader), 'Val MAE': val_mae/len(self.val_loader), 'Val MAPE': val_mape/len(self.val_loader)}) #/len(self.train_loader)})
            # test_losses.append(test_loss)
            print(f"Epoch {epoch} Val Losses: MSE {val_mse/len(self.val_loader)} | RMSE {val_rmse/len(self.val_loader)} | MAE {val_mae} | MAPE{val_mape/len(self.val_loader)}")
            
            if val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                torch.save(self.model.state_dict(), os.path.join(self.cfg['model']['save_path'], 'best_model.pt'))
                
                # torch.save({
                # 'epoch': epoch,
                # 'model_state_dict': self.model.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': min_val_loss,
                # }, self.cfg['model']['save_path'])
            
        torch.save(self.model.state_dict(), os.path.join(self.cfg['model']['save_path'], 'last_model.pt'))
        
        # plt.plot(train_losses)
        # plt.title('Train Loss')
        # plt.show()
        # plt.plot(val_losses)
        # plt.title('Validation Loss')
        # # plt.legend(['train_loss', 'val_loss'])
        # plt.show()
        # plt.plot(train_losses)
        # plt.plot(test_losses)
        # plt.legend(['epoch_loss', 'test_loss'])
        # plt.plot()
    
    def validate(self):
        pass
    
    def in_sample_prediction(self):
        y_true, y_preds = [], []
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch =  y_batch.to(self.device).squeeze(1)                                                                                    
                y_pred = self.model(X_batch.to(self.device)) #.detach().cpu()
                total_loss += self.loss_fn(y_pred, y_batch).item()
                y_true.append(y_batch.detach().cpu().numpy())
                y_preds.append(y_pred.detach().cpu().numpy())
        
        return np.stack(y_true), np.stack(y_preds), total_loss
