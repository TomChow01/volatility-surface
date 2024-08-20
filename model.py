import torch.nn as nn
import torch


hidden_dim = 64
n_features = 11


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cfg['model']['n_features'], hidden_size=cfg['model']['hidden_dim'], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.linear = nn.Linear(cfg['model']['hidden_dim'], cfg['model']['n_features']) 
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        # print('after LSTM',x.shape) 
        x = x[:, -1, :]  # (batch_size, n_lookback, hidden_dim) -> (batch_size, 1, hidden_dim)

        x = self.linear(x) #.view(-1,  n_forecast, input_dim)
        # print('after linear',x.shape)
        # print(x.shape)
        return x
    

class AttLSTM(nn.Module):
    def __init__(self, cfg):
        super(AttLSTM, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=cfg['model']['n_features'], hidden_size=cfg['model']['hidden_dim'], num_layers=1, batch_first=True)
        
        # Attention layer
        self.attention = nn.Linear(cfg['model']['hidden_dim'], 1)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=cfg['model']['hidden_dim'], hidden_size=cfg['model']['hidden_dim'], num_layers=1, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Fully connected layer
        self.fc = nn.Linear(cfg['model']['hidden_dim'], cfg['model']['n_features'])
    
    def forward(self, x):
        # x shape: (batch_size, t, cfg['model']['n_features'])
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        # lstm1_out shape: (batch_size, 3, 135)
        
        lstm1_out = self.dropout1(lstm1_out)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm1_out), dim=1)
        attended_out = lstm1_out * attention_weights
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(attended_out)
        # We only want the last time step output
        lstm2_out = lstm2_out[:, -1, :]
        # lstm2_out shape: (batch_size, 135)
        
        lstm2_out = self.dropout2(lstm2_out)
        
        # Fully connected layer
        out = self.fc(lstm2_out)
        # out shape: (batch_size, 45)
        
        return out