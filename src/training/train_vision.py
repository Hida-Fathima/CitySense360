from ultralytics import YOLO
import sys
import os
import shutil
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

MODEL_DIR = "models"

def train_vision_pro():
    print("🚀 Starting PROFESSIONAL Vision Training (GPU Enabled)...")
    
    # 1. Load MEDIUM Model
    
    model = YOLO('yolov8m.pt') 

    # 2. Train (With Overfitting Protection)
    results = model.train(
        data='data/vision/data.yaml', 
        epochs=50,          # Upper limit
        patience=10,        # SAFETY: Stops early if model stops learning
        imgsz=640,
        batch=16,           # Reduce to 8 if you get "Out of Memory"
        project=MODEL_DIR,
        name='yolo_road_pro',
        exist_ok=True,
        device=0,           # GPU
        optimizer='AdamW',
        lr0=0.001
    )

    # 3. Validate
    print("📊 Validating...")
    metrics = model.val()
    
    # 4. Save Final Weights
    src_path = os.path.join(MODEL_DIR, "yolo_road_pro", "weights", "best.pt")
    dst_path = os.path.join(MODEL_DIR, "road_damage_pro.pt")
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"✅ High-Acc Model saved: {dst_path}")
    
    # 5. Calculate "Dashboard Accuracy"
    # We use mAP@50 as the proxy for Accuracy Percentage
    map50 = float(metrics.box.map50)
    accuracy_pct = round(map50 * 100, 2)
    
    precision = float(round(metrics.box.mp, 4))
    recall = float(round(metrics.box.mr, 4))
    
    # 6. Log Metrics
    save_metrics(
        model_name="YOLOv8_Medium_Road",
        dataset_name="Pothole Detection v8",
        metrics={
            "Accuracy_Pct": accuracy_pct, 
            "mAP_50": round(map50, 4),
            "Precision": precision,
            "Recall": recall
        },
        params={"epochs": 50, "model": "yolov8m", "stopped_early": results.stop if hasattr(results, 'stop') else False},
        train_len="Auto",
        test_len="Auto"
    )
    
    print(f"🏆 Vision Training Complete. Dashboard Accuracy: {accuracy_pct}%")

if __name__ == "__main__":
    train_vision_pro()