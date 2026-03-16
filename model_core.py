import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# --- 1. 定義 LSTM 類別 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input_size 預設為 4 (對應 Return, Close, Volume, RSI)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])

# --- 2. 資料前處理與切分模組 ---
def prepare_model_data(df, look_back=60):
    """
    負責將 DataFrame 轉換為神經網路所需的特徵矩陣 (X) 與目標變數 (y)
    """
    # 提取特徵：報酬率、收盤價、成交量、RSI
    features = df[['Return', 'Close', 'Volume', 'RSI']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 為了避免未來函數 (Data Leakage)，Scaler 應該只 fit 在訓練集上 (前 80%)
    train_size = int(len(features) * 0.8)
    scaler.fit(features[:train_size])
    scaled_data = scaler.transform(features)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y.append(scaled_data[i, 0]) # 目標是預測 Return (第 0 個 column)

    return np.array(X), np.array(y), scaler, scaled_data

# --- 3. 訓練模型模組 ---
def train_lstm_model(X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    """
    封裝 PyTorch 的訓練迴圈
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(X_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).float().to(device)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for e in range(epochs):
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        
        # 提早停止機制 (Early Stopping)
        if epoch_loss < 0.0005:
            break
            
    return model, device

# --- 4. 預測與反轉指標模組 ---
def predict_model(model, X_test, device):
    """執行模型推論"""
    model.eval()
    X_tensor = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    return predictions

def get_inverse_price(pred_array, scaler, feature_count):
    """將模型預測的 Scaled 數值還原回真實報酬率"""
    dummy = np.zeros((len(pred_array), feature_count))
    dummy[:, 0] = pred_array.flatten()
    return scaler.inverse_transform(dummy)[:, 0]