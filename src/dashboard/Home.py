import streamlit as st
import os
import sys
import requests # Added for API calls

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CitySense360 | Integrated Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILS ---
def check_model_status(path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)

def get_live_weather():
    """Fetches live weather for Lat 15.2993, Long 74.1240 (Goa/Karnataka Border)"""
    try:
        # Enhanced URL to get Current Temp, Rain, Wind, and Humidity
        url = "https://api.open-meteo.com/v1/forecast?latitude=15.2993&longitude=74.1240&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m&timezone=auto"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

def get_weather_emoji(code):
    """Maps WMO codes to Emojis"""
    if code == 0: return "☀️ Clear Sky"
    if code in [1, 2, 3]: return "⛅ Partly Cloudy"
    if code in [45, 48]: return "🌫️ Foggy"
    if code in [51, 53, 55, 61, 63, 65]: return "🌧️ Rainy"
    if code in [71, 73, 75]: return "❄️ Snowy"
    if code in [95, 96, 99]: return "⛈️ Thunderstorm"
    return "🌥️ Overcast"

# --- HERO SECTION ---
st.title("🏙️ CitySense360")
st.markdown("### **AI-Powered Smart City Intelligence & Public Infrastructure Automation**")

# --- LIVE WEATHER WIDGET ---
weather_data = get_live_weather()

if weather_data and 'current' in weather_data:
    curr = weather_data['current']
    
    # Custom CSS container for weather
    st.markdown("""
    <style>
    .weather-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #90caf9;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"**📍 Live Environment Status (Location: 15.29°N, 74.12°E)**")
        w1, w2, w3, w4 = st.columns(4)
        
        with w1:
            st.metric("Temperature", f"{curr['temperature_2m']}°C", get_weather_emoji(curr['weather_code']))
        with w2:
            st.metric("Precipitation", f"{curr['precipitation']} mm", "Real-time")
        with w3:
            st.metric("Humidity", f"{curr['relative_humidity_2m']}%", "Air Quality Impact")
        with w4:
            st.metric("Wind Speed", f"{curr['wind_speed_10m']} km/h", "Sensor Readings")
        
        st.divider()
else:
    st.warning("⚠️ Weather API Unreachable (Check Internet Connection)")

st.markdown("""
<div style='background-color: #F0F2F6; padding: 20px; border-radius: 10px; border-left: 5px solid #0052CC;'>
    <strong>Project Overview:</strong> CitySense360 is a unified Urban Intelligence Platform integrating 
    <strong>Computer Vision, NLP, Deep Learning, RAG and Agentic AI</strong>. 
    It automates civic operations ranging from <em>Ridership Forecasting</em> and <em>Infrastructure Defect Detection</em> 
    to <em>Automated Grievance Redressal</em> and <em>Citizen Support</em>.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- REAL-TIME AI ENGINE STATUS (All 7 Models) ---
st.subheader("🛠️ AI Engine Health Check")

# Row 1: Time Series & Forecasting
c1, c2, c3 = st.columns(3)
with c1:
    if check_model_status("models/metro_lstm.pt"):
        st.success("🟢 **Metro Analytics**\n\n*Online (Bi-LSTM)*")
    else:
        st.error("🔴 **Metro Analytics**\n\n*Offline*")

with c2:
    if check_model_status("models/traffic_lstm.pt"):
        st.success("🟢 **Traffic Forecaster**\n\n*Online (Bi-LSTM)*")
    else:
        st.error("🔴 **Traffic Forecaster**\n\n*Offline*")

with c3:
    if check_model_status("models/energy_gru.pt"):
        st.success("🟢 **Energy Monitor**\n\n*Online (GRU)*")
    else:
        st.error("🔴 **Energy Monitor**\n\n*Offline*")

# Row 2: Vision & Environment
c4, c5, c6 = st.columns(3)
with c4:
    if check_model_status("models/safety_classifier.pt"):
        st.success("🟢 **Safety Vision**\n\n*Online (YOLOv8-Cls)*")
    else:
        st.error("🔴 **Safety Vision**\n\n*Offline*")

with c5:
    if check_model_status("models/road_damage_pro.pt"):
        st.success("🟢 **Road Inspector**\n\n*Online (YOLOv8)*")
    else:
        st.error("🔴 **Road Inspector**\n\n*Offline*")

with c6:
    if check_model_status("models/pollution_xgb.json"):
        st.success("🟢 **Pollution Tracker**\n\n*Online (XGBoost)*")
    else:
        st.warning("🟡 **Pollution Tracker**\n\n*Pending Training*")

# Row 3: Language Intelligence
c7, c8, c9 = st.columns(3)
with c7:
    if check_model_status("models/complaints_bert/config.json"):
        st.success("🟢 **Grievance Agent**\n\n*Online (DistilBERT + Llama3)*")
    else:
        st.error("🔴 **Grievance Agent**\n\n*Offline*")

with c8:
    if check_model_status("models/vector_store/faiss_index/index.faiss"):
        st.success("🟢 **Public Help Bot**\n\n*Online (RAG + Llama3)*")
    else:
        st.warning("🟡 **Public Help Bot**\n\n*Re-Ingest Required*")

with c9:
    st.info("🔵 **System Core**\n\n*PyTorch + LangChain + Streamlit*")

st.divider()

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 50px;'>
    <hr>
    <p>CitySense360 | Status: Development | GPU Acceleration: Enabled</p>
    <h3>Made by Hida Fathima</h3>
</div>
""", unsafe_allow_html=True)