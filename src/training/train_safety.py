from ultralytics import YOLO
import sys
import os
import shutil

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

def train_safety_classification():
    print("🚨 Starting Accident Classification Training (GPU)...")
    
    # 1. Load the Classification Model (Note the '-cls' suffix)
    # We use 'yolov8m-cls.pt' which is optimized for classifying whole images.
    model = YOLO('yolov8m-cls.pt') 

    # 2. Train
    # We point directly to the folder containing train/test/val
    results = model.train(
        data='data/safety',  # YOLO scans this folder for class names automatically
        epochs=20,           # Classification learns fast, 20 epochs is usually enough
        imgsz=224,           # Standard size for classification
        batch=32,            # Higher batch size since classification is lighter
        device=0,            # GPU
        project='models',
        name='yolo_safety_cls',
        exist_ok=True
    )

    # 3. Validate
    print("📊 Validating...")
    metrics = model.val()
    
    # 4. Save Best Model
    # YOLO saves the best run in 'models/yolo_safety_cls/weights/best.pt'
    src_path = "models/yolo_safety_cls/weights/best.pt"
    dst_path = "models/safety_classifier.pt"
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"✅ Safety Model Saved: {dst_path}")

    # 5. Log Metrics
    # Classification metrics are Top-1 and Top-5 Accuracy
    top1_acc = metrics.top1
    top5_acc = metrics.top5

    save_metrics(
        model_name="YOLOv8m-Cls_Accident",
        dataset_name="CCTV Accident Classification",
        metrics={
            "Top1_Accuracy": round(top1_acc * 100, 2),
            "Top5_Accuracy": round(top5_acc * 100, 2)
        },
        params={"epochs": 20, "imgsz": 224, "type": "Classification"},
        train_len="Auto",
        test_len="Auto"
    )
    print(f"🏆 Training Complete. Accuracy: {top1_acc*100:.2f}%")

if __name__ == "__main__":
    train_safety_classification()