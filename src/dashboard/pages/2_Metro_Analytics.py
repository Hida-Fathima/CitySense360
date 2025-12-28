import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
import os
import sys

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# Importing class and constants from your training script
from src.training.train_transport import MetroLSTM, HIDDEN_SIZE, LAYERS 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Metro Analytics", page_icon="🚇", layout="wide")

# --- CONSTANTS ---
MODEL_PATH = "models/metro_lstm.pt"
SCALER_PATH = "models/metro_scaler.pkl"
DATA_PATH = "data/transport/delhi_metro.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 30 # Must match the lookback window used in training

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load Scaler
    if not os.path.exists(SCALER_PATH):
        return None, None
    scaler = joblib.load(SCALER_PATH)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        return None, scaler
    
    # Initialize the model structure
    model = MetroLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS).to(DEVICE)
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return model, scaler

model, scaler = load_resources()

# --- HEADER ---
st.title("🚇 Delhi Metro Command Center")
st.markdown("### *AI-Powered Ridership Forecasting & Operations*")
st.markdown("---")

# --- DATA PROCESSING ---
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Data missing: {DATA_PATH}. Please ensure your Delhi Metro CSV is in the data/transport/ folder.")
    st.stop()

# Load and aggregate data for visualization
df = pd.read_csv(DATA_PATH)
date_col = 'Date' if 'Date' in df.columns else 'DateTime'
df[date_col] = pd.to_datetime(df[date_col])
# Aggregating all transactions into daily totals
daily_df = df.groupby(date_col)['Passengers'].sum().reset_index().sort_values(date_col)

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
last_day_passengers = daily_df['Passengers'].iloc[-1]
avg_weekly = daily_df['Passengers'].tail(7).mean()

with col1:
    st.metric("Daily Passengers", f"{last_day_passengers:,.0f}", "Last 24h")
with col2:
    st.metric("7-Day Average", f"{avg_weekly:,.0f}", "Trend")
with col3:
    st.metric("Model Accuracy", "92.91%", "Bi-LSTM (GPU)")
with col4:
    st.metric("System Status", "Operational", "Normal", delta_color="normal")

# --- FORECASTING LOGIC ---
st.subheader("📈 Ridership Trends & AI Forecast")

if model and scaler:
    # Prepare Data for Prediction
    data_values = daily_df['Passengers'].values.reshape(-1, 1)
    data_scaled = scaler.transform(data_values)
    
    # Predict Next 7 Days using a recursive sliding window
    future_preds = []
    
    # FIX: Prepare initial batch as Tensor and move to GPU
    current_batch_np = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    current_batch = torch.tensor(current_batch_np, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(7): # Forecast for 1 week
            # Pass current window into Bi-LSTM
            pred = model(current_batch)
            future_preds.append(pred.cpu().item())
            
            # Update the sliding window: 
            # 1. Remove the oldest day [:, 1:, :]
            # 2. Append the new prediction [new_pred_reshaped]
            new_pred_reshaped = pred.reshape(1, 1, 1)
            current_batch = torch.cat((current_batch[:, 1:, :], new_pred_reshaped), dim=1)

    # Inverse Transform predictions back to real numbers
    future_vals = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    
    # Create Future Dates for the X-axis
    last_date = daily_df[date_col].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    # --- PLOTTING ---
    fig = go.Figure()
    
    # Historical Data (Last 60 Days for clarity)
    plot_df = daily_df.tail(60)
    fig.add_trace(go.Scatter(
        x=plot_df[date_col], 
        y=plot_df['Passengers'],
        mode='lines',
        name='Historical Ridership',
        line=dict(color='#00CC96', width=2)
    ))
    
    # AI Forecast
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_vals.flatten(),
        mode='lines+markers',
        name='AI Prediction (7 Days)',
        line=dict(color='#EF553B', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Metro Network Traffic Demand",
        xaxis_title="Date",
        yaxis_title="Total Passengers",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight Card
    peak_val = int(future_vals.max())
    peak_day = future_dates[np.argmax(future_vals)].strftime('%A')
    st.info(f"💡 **AI Predictive Insight:** Weekly demand is expected to peak at **{peak_val:,.0f}** passengers on **{peak_day}**. Deployment of additional train sets is recommended during peak hours.")

else:
    st.warning("⚠️ Model not loaded. Please ensure training artifacts (metro_lstm.pt and metro_scaler.pkl) exist in the models/ folder.")

# --- ANALYTICS BREAKDOWN ---
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("🎫 Ticketing Patterns")
    if 'Ticket_Type' in df.columns:
        ticket_counts = df['Ticket_Type'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=ticket_counts.index, values=ticket_counts.values, hole=.3)])
        fig_pie.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("Detailed ticket metadata not found in CSV.")

with c2:
    st.subheader("📍 Station Congestion Levels")
    # Identify high-demand stations from either From_Station or To_Station
    if 'From_Station' in df.columns:
        top_stations = df['From_Station'].value_counts().head(10)
        fig_bar = go.Figure(go.Bar(
            x=top_stations.values, 
            y=top_stations.index, 
            orientation='h',
            marker=dict(color='#636EFA')
        ))
        fig_bar.update_layout(template="plotly_dark", xaxis_title="Transaction Volume", yaxis_autorange="reversed")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.caption("Station-level analytics unavailable.")

st.markdown("---")
st.caption("Data Source: Delhi Metro Ridership Dataset (2022-2024) | Inference Engine: PyTorch / CUDA Enabled")