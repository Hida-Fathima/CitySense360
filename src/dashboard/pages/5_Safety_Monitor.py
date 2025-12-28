import streamlit as st
import torch
from ultralytics import YOLO
import os
from PIL import Image

st.set_page_config(page_title="Safety Monitor", page_icon="🚨", layout="wide")

MODEL_PATH = "models/safety_classifier.pt"

st.title("🚨 City Safety Monitor")
st.markdown("### *Real-time Accident & Hazard Detection (YOLOv8)*")
st.divider()

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    st.sidebar.success(f"✅ Safety Model Loaded ({MODEL_PATH})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📸 **Upload CCTV Frame**")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Surveillance Feed", use_container_width=True)
            
            # Inference
            results = model(img)
            
            # Display Results
            probs = results[0].probs
            top1_index = probs.top1
            label = results[0].names[top1_index]
            conf = probs.top1conf.item()
            
            with col2:
                st.subheader("🔍 AI Analysis")
                if label.lower() in ['accident', 'fire', 'crash']:
                    st.error(f"⚠️ **DANGER DETECTED: {label.upper()}**")
                else:
                    st.success(f"✅ **Status: {label.upper()}**")
                
                st.metric("Confidence Score", f"{conf*100:.1f}%")
                st.progress(conf)

else:
    st.error(f"❌ Safety Model not found at {MODEL_PATH}")