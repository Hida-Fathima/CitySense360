import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
import os
import torch.nn as nn

# --- PAGE CONFIG ---
st.set_page_config(page_title="Traffic Forecaster", page_icon="🚦", layout="wide")

# --- CONSTANTS ---
MODEL_PATH = "models/traffic_lstm.pt"
SCALER_PATH = "models/traffic_scaler.pkl"
DATA_PATH = "data/traffic/traffic_clean.csv" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 24  # Matched to training script

# --- MODEL DEFINITION (MATCHES CHECKPOINT) ---
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2): 
        super(TrafficLSTM, self).__init__()
        
        # Bidirectional must be True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True)
        
        # FC Input = hidden_size * 2 (128 * 2 = 256)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- LOAD RESOURCES ---
@st.cache_resource
def load_traffic_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    
    scaler = joblib.load(SCALER_PATH)
    model = TrafficLSTM().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None
        
    model.eval()
    return model, scaler

model, scaler = load_traffic_model()

# --- HEADER ---
st.title("🚦 Smart Traffic Forecast")
st.markdown("### *Congestion Prediction Engine (Bi-LSTM)*")
st.divider()

# --- MAIN LOGIC ---
if model and scaler:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        # --- 1. SMART COLUMN DETECTION ---
        # Convert columns to lowercase for easy searching
        cols_lower = [c.lower() for c in df.columns]
        
        # A. Find Date Column
        date_col = None
        if 'date_time' in cols_lower: date_col = df.columns[cols_lower.index('date_time')]
        elif 'date' in cols_lower: date_col = df.columns[cols_lower.index('date')]
        elif 'timestamp' in cols_lower: date_col = df.columns[cols_lower.index('timestamp')]
        
        # B. Find Traffic Column (Target)
        target_col = None
        possible_targets = ['traffic_volume', 'vehicle_count', 'count', 'traffic', 'volume']
        for t in possible_targets:
            if t in cols_lower:
                target_col = df.columns[cols_lower.index(t)]
                break
        
        # Fallback: Use the last numeric column that isn't Year/ID
        if target_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if 'year' not in c.lower() and 'id' not in c.lower()]
            if numeric_cols:
                target_col = numeric_cols[-1]
            else:
                st.error("❌ Could not find a numeric 'Traffic Volume' column in CSV.")
                st.stop()

        # --- 2. PREPARE DATA ---
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col)
            dates = df[date_col]
        else:
            dates = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

        # --- 3. VISUALIZE HISTORY ---
        st.subheader("📉 Traffic Density Trends")
        fig = go.Figure()
        
        # Plot last 100 hours of REAL data
        fig.add_trace(go.Scatter(
            x=dates.tail(100), 
            y=df[target_col].tail(100), 
            mode='lines', 
            name='Actual History', 
            line=dict(color='#FFA500', width=2)
        ))
        
        # --- 4. RUN PREDICTION (Next 24 Hours) ---
        raw_seq = df[target_col].values[-SEQ_LENGTH:].reshape(-1, 1)
        last_seq_scaled = scaler.transform(raw_seq)
        current_batch = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        future_preds = []
        with torch.no_grad():
            for _ in range(24): # Predict 24 hours
                pred = model(current_batch)
                future_preds.append(pred.item())
                new_pred_reshaped = pred.reshape(1, 1, 1)
                current_batch = torch.cat((current_batch[:, 1:, :], new_pred_reshaped), dim=1)
                
        # Inverse transform to get real vehicle numbers
        future_real = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        
        # Generate Future Dates
        last_date = dates.iloc[-1]
        future_dates = [last_date + pd.Timedelta(hours=i) for i in range(1, 25)]
        
        # Plot PREDICTIONS
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_real.flatten(), 
            mode='lines+markers', 
            name='AI Forecast (24h)', 
            line=dict(color='#00FF00', width=3, dash='dot')
        ))
        
        fig.update_layout(
            template="plotly_white", 
            height=500, 
            xaxis_title="Time", 
            yaxis_title="Vehicles per Hour",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 5. METRICS & INSIGHTS ---
        c1, c2 = st.columns(2)
        peak_traffic = int(np.max(future_real))
        
        if peak_traffic > 4500:
            status = "🔴 Heavy Congestion"
            insight = "Traffic is expected to exceed road capacity. Suggest activating diversion protocols."
        elif peak_traffic > 3000:
            status = "🟡 Moderate Flow"
            insight = "Traffic flow is steady but high. Signal optimization recommended."
        else:
            status = "🟢 Free Flow"
            insight = "Road utilization is optimal. No intervention needed."

        c1.metric("Predicted Peak Traffic", f"{peak_traffic} Vehicles", status)
        c2.info(f"💡 **AI Insight:** {insight}")
        
    else:
        st.error(f"❌ Data file not found: {DATA_PATH}. Check 'data/traffic/' folder.")
else:
    st.warning("⚠️ Model not loaded correctly. Check the error above.")