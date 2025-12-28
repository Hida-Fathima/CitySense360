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

# --- CONFIGURATION ---
SEQ_LENGTH = 24     # Look back 24 hours
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
HIDDEN_SIZE = 64
LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🇮🇳 DATASET PATH
DATA_PATH = "data/energy/india_monthly_electricity.csv"
MODEL_PATH = "models/energy_gru.pt"
SCALER_PATH = "models/energy_scaler.pkl"

class EnergyGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS):
        super(EnergyGRU, self).__init__()
        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
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

def process_monthly_data(df):
    """
    1. Filters for Karnataka (Bangalore).
    2. SUMS all fuel sources (Coal + Solar + Wind) to get TOTAL Generation.
    3. Upsamples to Hourly.
    """
    print("   Processing Indian Energy Data...")
    
    # 1. Filter: Karnataka (Bangalore) AND 'Electricity generation'
    # We ignore 'Emissions' and 'Capacity' for the GRU model, we only want actual Generation (GWh)
    mask = (df['State'] == "Karnataka") & (df['Category'] == "Electricity generation")
    df_state = df[mask].copy()
    
    if df_state.empty:
        print("   ⚠️ Karnataka data not found. Switching to 'India' (Total).")
        mask = (df['State'] == "India") & (df['Category'] == "Electricity generation")
        df_state = df[mask].copy()

    # 2. Parse Dates (Your file uses YYYY-MM-DD standard format)
    df_state['Date'] = pd.to_datetime(df_state['Date'])
    
    # 3. Pivot & Sum
    # We want to sum all variables (Coal, Hydro, Solar, etc.) for each Date to get TOTAL Grid Load
    # We explicitly drop NaNs to avoid errors
    df_grouped = df_state.groupby('Date')['Value'].sum().reset_index()
    
    # 4. Set Index & Sort
    df_grouped = df_grouped.sort_values('Date').set_index('Date')
    
    # 5. Upsample to Hourly ('H') and Interpolate
    # This creates the smooth curve needed for Deep Learning from monthly points
    df_hourly = df_grouped['Value'].resample('h').interpolate(method='quadratic')
    
    # Handle negatives (interpolation sometimes dips below 0)
    df_hourly = df_hourly.clip(lower=0)
    
    # Convert series to numpy array
    return df_hourly.values.reshape(-1, 1)

def train_energy_gru():
    print(f"⚡ Starting Indian Energy GRU Training on {DEVICE}...")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Data missing. Check 'data/energy/india_monthly_electricity.csv'")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        data_values = process_monthly_data(df)
        print(f"   Generated {len(data_values)} hourly data points (Sum of all Sources).")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # 2. Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # 3. Create Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # 4. Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 5. Train
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    model = EnergyGRU().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("   Training...")
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
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
        mae = mean_absolute_error(y_test_actual, preds_actual)
        r2 = r2_score(y_test_actual, preds_actual)
        
        mean_val = np.mean(y_test_actual)
        accuracy_pct = max(0, 1.0 - (mae / mean_val)) * 100

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    save_metrics(
        model_name="India_Energy_GRU",
        dataset_name="India Electricity (Summed)",
        metrics={
            "RMSE": float(round(rmse, 2)),
            "Accuracy_Pct": float(round(accuracy_pct, 2)),
            "R2_Score": float(round(r2, 4))
        },
        params={"type": "GRU", "layers": LAYERS, "hidden": HIDDEN_SIZE},
        train_len=len(X_train),
        test_len=len(X_test)
    )
    print(f"✅ Indian Energy Model Saved. Accuracy: {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_energy_gru()