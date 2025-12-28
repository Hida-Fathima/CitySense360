import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os

st.set_page_config(page_title="Pollution Tracker", page_icon="🌫️", layout="wide")

MODEL_PATH = "models/pollution_xgb.json"
# FIX: Updated to match your file structure image
DATA_PATH = "data/pollution/indian_aqi.csv" 

st.title("🌫️ Urban Pollution Tracker")
st.markdown("### *Hyperlocal Air Quality Forecasting (XGBoost)*")
st.divider()

# Load Model
if os.path.exists(MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    st.sidebar.success("✅ XGBoost Model Loaded")
else:
    st.sidebar.error("❌ Model Missing")

# Load Data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    # Auto-detect AQI column
    aqi_col = 'AQI' if 'AQI' in df.columns else df.columns[-1]
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg AQI", int(df[aqi_col].mean()), "Normal")
    c2.metric("Data Points", f"{len(df)}", "City Sensors")
    c3.metric("Primary Pollutant", "PM 2.5")

    # Chart
    st.subheader("📊 Air Quality Trends")
    
    # Create a simple index for plotting if Date is missing
    if 'Date' in df.columns:
        x_axis = 'Date'
    else:
        df['Index'] = range(len(df))
        x_axis = 'Index'
        
    fig = px.line(df.tail(200), x=x_axis, y=aqi_col, title="Pollutant Levels Over Time")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.error(f"❌ Data missing: {DATA_PATH}")