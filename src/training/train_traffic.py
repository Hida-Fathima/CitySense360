import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

# --- CONFIG (Upgraded Internals) ---
SEQ_LENGTH = 24
EPOCHS = 100         # Silent Upgrade: More training
BATCH_SIZE = 32
LR = 0.001
HIDDEN_SIZE = 128    # Silent Upgrade: Bigger Brain
LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/traffic/traffic_clean.csv"
MODEL_PATH = "models/traffic_lstm.pt"       
SCALER_PATH = "models/traffic_scaler.pkl"  

# --- THE MODEL (Bi-LSTM)---
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS):
        super(TrafficLSTM, self).__init__()
        # We use bidirectional=True for better accuracy, but keep class name same
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        # Input * 2 because Bi-Directional outputs 2 vectors
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def train_traffic():
    print(f"🚦 Starting Traffic Model Training on {DEVICE}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data missing at {DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Auto-detect traffic volume column
    if 'traffic_volume' in df.columns:
        data = df['traffic_volume'].values.reshape(-1, 1)
    else:
        # Fallback: Use last numeric column
        data = df.select_dtypes(include=[np.number]).iloc[:, -1].values.reshape(-1, 1)
        
    # 2. Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 3. Create Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # 4. Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 5. Train
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    model = TrafficLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.5f}")

    # 6. Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(X_test_tensor).cpu().numpy()
        
        preds_actual = scaler.inverse_transform(preds)
        y_test_actual = scaler.inverse_transform(y_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
        mae = mean_absolute_error(y_test_actual, preds_actual)
        r2 = r2_score(y_test_actual, preds_actual)
        
        # Calculate Accuracy %
        mean_val = np.mean(y_test_actual)
        accuracy_pct = max(0, 1.0 - (mae / mean_val)) * 100

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Log "Traffic_LSTM" so Dashboard updates automatically
    save_metrics(
        model_name="Traffic_LSTM",
        dataset_name="City Traffic Cleaned",
        metrics={
            "RMSE": float(round(rmse, 2)),
            "Accuracy_Pct": float(round(accuracy_pct, 2)),
            "R2_Score": float(round(r2, 4))
        },
        params={"type": "Bi-LSTM (Upgraded)", "epochs": EPOCHS, "hidden": HIDDEN_SIZE},
        train_len=len(X_train),
        test_len=len(X_test)
    )
    print(f"✅ Traffic Model Updated. New Accuracy: {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_traffic()