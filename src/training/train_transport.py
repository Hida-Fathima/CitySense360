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

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

# --- CONFIG ---
SEQ_LENGTH = 30   # Look back at the last 30 days
EPOCHS = 100      # Higher epochs for GPU
BATCH_SIZE = 32
HIDDEN_SIZE = 128 # Bigger brain for complex metro data
LAYERS = 2
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_PATH = "data/transport/delhi_metro.csv"
MODEL_PATH = "models/metro_lstm.pt"
SCALER_PATH = "models/metro_scaler.pkl"

# --- NEURAL NETWORK ---
class MetroLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS):
        super(MetroLSTM, self).__init__()
        # Bidirectional = True allows the model to see context from both past and future during training
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        # Output layer (Hidden * 2 because of bidirectional)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step
        out = out[:, -1, :]
        return self.fc(out)

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

def process_metro_data():
    """
    Reads the Delhi Metro CSV and aggregates it into a Daily Time-Series.
    """
    print("🚇 Processing Delhi Metro Data...")
    
    try:
        # Load CSV
        df = pd.read_csv(DATA_PATH)
        
        # 1. Standardize Date Column
        # Dataset has 'Date' or 'DateTime'
        date_col = 'Date' if 'Date' in df.columns else 'DateTime'
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 2. Group by Date to get Total Daily Passengers
        # We sum up all passengers from all routes for that day
        daily_df = df.groupby(date_col)['Passengers'].sum().reset_index()
        daily_df = daily_df.sort_values(date_col)
        
        print(f"   ✓ Aggregated into {len(daily_df)} days of data.")
        print(f"   ✓ Date Range: {daily_df[date_col].min()} to {daily_df[date_col].max()}")
        
        return daily_df['Passengers'].values.reshape(-1, 1)

    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None

def train_transport():
    print(f"🚄 Starting Delhi Metro Training on {DEVICE}...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Data missing at {DATA_PATH}")
        return

    # 1. Get Data
    data_values = process_metro_data()
    if data_values is None: return

    # 2. Scale (Normalization is crucial for LSTMs)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_values)

    # 3. Create Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)

    # 4. Split (80% Train, 20% Test)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # 5. Dataloaders
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Initialize Model
    model = MetroLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 7. Training Loop
    print(f"   Training for {EPOCHS} epochs...")
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
            print(f"   Epoch {epoch+1}: Loss {epoch_loss/len(train_loader):.6f}")

    # 8. Evaluation
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(test_input).cpu().numpy()
        
        # Inverse Transform to get real passenger numbers
        preds_actual = scaler.inverse_transform(preds)
        y_test_actual = scaler.inverse_transform(y_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
        mae = mean_absolute_error(y_test_actual, preds_actual)
        r2 = r2_score(y_test_actual, preds_actual)
        
        # Accuracy Proxy
        mean_val = np.mean(y_test_actual)
        accuracy_pct = max(0, 1.0 - (mae / mean_val)) * 100

    # 9. Save Artifacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # 10. Log Metrics
    save_metrics(
        model_name="Delhi_Metro_LSTM",
        dataset_name="Delhi Metro 2022-24",
        metrics={
            "RMSE": float(round(rmse, 2)),
            "Accuracy_Pct": float(round(accuracy_pct, 2)),
            "R2_Score": float(round(r2, 4))
        },
        params={"type": "Bi-LSTM", "lookback": SEQ_LENGTH, "hidden": HIDDEN_SIZE},
        train_len=len(X_train),
        test_len=len(X_test)
    )
    print(f"🏆 Metro Model Saved. Accuracy: {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_transport()