import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Road Inspector", page_icon="🛣️", layout="wide")

MODEL_PATH = "models/road_damage_pro.pt"

st.title("🛣️ AI Road Inspector")
st.markdown("### *Automated Pothole & Crack Detection (YOLOv8)*")
st.divider()

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    st.sidebar.success("✅ Road Damage Model Loaded")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Road Image")
        # FIX: Added 'webp' to allow your test images
        uploaded_file = st.file_uploader("Upload an image of the road...", type=['jpg', 'png', 'jpeg', 'webp'])
        
        if uploaded_file:
            # Display Original
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Defect Detection")
                with st.spinner("Analyzing road surface..."):
                    # Run Inference
                    results = model(image)
                    res_plotted = results[0].plot() # Draws boxes
                    
                    # Display Result
                    st.image(res_plotted, caption="AI Detection Result", use_container_width=True)
                    
                    # Detection Metrics
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.error(f"⚠️ Found {len(boxes)} Road Defects!")
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            try:
                                cls_name = model.names[cls_id]
                            except:
                                cls_name = "Defect"
                            conf = float(box.conf[0])
                            st.write(f"- **{cls_name.upper()}** (Confidence: {conf*100:.1f}%)")
                    else:
                        st.success("✅ No defects detected. Road is healthy.")
else:
    st.warning(f"⚠️ Model not found at {MODEL_PATH}. Please train the YOLO model first.")