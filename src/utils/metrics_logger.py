import json
import os
from datetime import datetime

# Define path
METRICS_DIR = "metrics"

def save_metrics(model_name, dataset_name, metrics, params, train_len, test_len):
    """
    Saves metrics ONLY to the '_latest.json' file.
    Overwrites previous runs to keep the folder clean.
    """
    # Ensure metrics directory exists
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR)

    # Create the data structure
    data = {
        "model_name": model_name,
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_stats": {
            "train_samples": train_len,
            "test_samples": test_len
        },
        "metrics": metrics,
        "parameters": params
    }

    # Save ONLY the "Latest" file (Overwrites previous data)
    filename = f"{model_name.lower()}_latest.json"
    latest_path = os.path.join(METRICS_DIR, filename)
    
    with open(latest_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Metrics updated: {latest_path}")