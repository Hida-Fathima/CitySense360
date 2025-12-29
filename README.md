# 🏙️ CitySense360: AI-Powered Smart City Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-00FFFF?logo=opencv&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agentic%20AI-1C3C3C?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

**CitySense360** is a unified Urban Intelligence Dashboard integrating **Computer Vision, Deep Learning (LSTM/GRU), and Generative AI (RAG agents)**. It automates civic operations ranging from real-time traffic forecasting and infrastructure defect detection to automated citizen grievance redressal.

---

## 🚀 Key Modules & Capabilities

### 🧠 1. Agentic AI & NLP

* **📢 Automated Grievance Agent:** A specialized AI Agent (DistilBERT + Llama 3) that classifies citizen complaints into 81 departments, estimates severity, and drafts official "Internal Work Orders" and "Citizen Responses" automatically.
* **🤖 Public Help Bot (RAG):** A Retrieval-Augmented Generation chatbot that answers queries on traffic rules, waste management, and disaster safety by referencing official government PDFs (Motor Vehicles Act, Swachh Bharat Guidelines).

### 👁️ 2. Computer Vision (YOLOv8)

* **🛣️ AI Road Inspector:** Detects potholes and road cracks in real-time using a fine-tuned **YOLOv8m** model (mAP@50: 81.3%).
* **🚨 Safety Monitor:** Analyzes CCTV frames to detect accidents, fire, and safety hazards with **94.9% accuracy**.

### 📈 3. Predictive Analytics (Time-Series)

* **🚦 Traffic Forecaster:** Uses **Bi-Directional LSTMs** to predict traffic volume 24 hours ahead based on historical congestion patterns (Accuracy: 92.5%).
* **🚇 Metro Analytics:** Forecasts daily ridership for public transport optimization using **Bi-LSTM** (Accuracy: 92.9%).
* **⚡ Energy Monitor:** Predicts city-wide power consumption loads using **GRU (Gated Recurrent Units)** (Accuracy: 95.6%).
* **🌫️ Pollution Tracker:** Hyperlocal Air Quality (AQI) forecasting using **XGBoost** with geography-aware feature engineering.

---

## 🛠️ Tech Stack

* **Core Framework:** Python 3.10, Streamlit
* **Deep Learning:** PyTorch (CUDA Optimized), TensorFlow/Keras
* **Computer Vision:** Ultralytics YOLOv8, OpenCV, PIL
* **LLM & Agents:** LangChain, Ollama (Llama 3), FAISS Vector DB, Hugging Face Transformers
* **Data Science:** Pandas, NumPy, Scikit-Learn, Plotly, XGBoost
* **Deployment:** Docker, Containerized Environment

---

## 🚀 Deployment & Installation

### Option 1: Full Version (Docker) - Recommended
*Includes Generative AI Agents (RAG), Chatbots & Predictive Analytics.*
**Prerequisites:** [Docker Desktop](https://www.docker.com/) and [Ollama](https://ollama.com/) running locally.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Hida-Fathima/CitySense360.git
   cd CitySense360
   ```
2. **Build the container:**

   ```bash
   docker build -t citysense360 .
   ```
3. **Run the app:**

   ```bash
   docker run -p 8501:8501 citysense360
   ```
*Note: Ensure your local Ollama server is running for AI features.*


### Option 2: Lite Version (Streamlit Cloud)

Includes Analytics Dashboard & Forecasting modules only. AI Agents are disabled.

**Live Demo:** 🔗 Click Here to View Live App (Replace this with your actual link later)

**Branch:** `cloud-lite`

**Features:** Metro, Traffic, Energy, and Pollution analytics are fully operational.

*Note: RAG Agents are disabled in this version due to cloud resource limits.*


### Option 3: Full Version (Streamlit local)

### 1. Clone the Repository

```bash
git clone https://github.com/Hida-Fathima/CitySense360.git
cd CitySense360
```


### 2. Install Dependencies

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 3. Setup LLM (Ollama)

This project uses **Llama 3** locally for privacy and cost-efficiency.

- Download Ollama
- Pull the Llama 3 model:

```bash
ollama pull llama3
```

### 4. Run the Dashboard

```bash
streamlit run src/dashboard/Home.py
```


## 📂 Dataset Information

To keep the repository lightweight, raw training images (5GB+) are excluded. Inference runs out-of-the-box using the pre-trained models included in `models/`.

**Sources for Retraining:**

* **Traffic:** [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
* **Road Defects:** [Kaggle Pothole Detection (YOLOv8)](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)
* **Energy:** [India Monthly Electricity Consumption](https://www.kaggle.com/datasets/rhtsingh/india-monthly-electricity-consumption-20192025)
* **Complaints:** [Govt of India Grievance Data](https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report/data)
* **Safety/CCTV:** [Accident Detection Footage](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage)

---

## 📊 Model Performance Metrics

| Module | Model Architecture | Metric | Score |
| --- | --- | --- | --- |
| **Traffic** | Bi-Directional LSTM | Accuracy | **92.52%** |
| **Metro** | Bi-Directional LSTM | Accuracy | **92.91%** |
| **Energy** | GRU (2 Layers) | Accuracy | **95.65%** |
| **Safety** | YOLOv8m-Cls | Top-1 Acc | **94.90%** |
| **Road Defects** | YOLOv8m | mAP@50 | **81.33%** |
| **Complaints** | DistilBERT | Accuracy | **84.30%** |

---

## 🐳 Docker Support

This application is fully containerized and Docker-ready. It utilizes a custom `Dockerfile` to handle the complex dependency chain involving PyTorch, CUDA, and Streamlit, ensuring consistent performance across different deployment environments.
***