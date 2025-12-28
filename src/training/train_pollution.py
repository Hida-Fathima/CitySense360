import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os
import sys

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

# --- CONFIG ---
DATA_PATH = "data/pollution/indian_aqi.csv"
MODEL_PATH = "models/pollution_xgb.json"
SCALER_PATH = "models/pollution_scaler.pkl"

def process_geo_data(df):
    print("   Detected 'Long Format'. Pivoting & Keeping Geography...")
    # Pivot but keep 'state' in the index so we can use it as a feature
    df_pivot = df.pivot_table(
        index=['state', 'city', 'station'], 
        columns='pollutant_id', 
        values='pollutant_avg',
        aggfunc='mean'
    ).reset_index()
    return df_pivot

def augment_continuous(X, y):
    """
    Augments ONLY the continuous features (Chemistry), not the Categorical ones (States).
    """
    print(f"   ⚗️ Augmenting Data: {len(X)} -> {len(X)*2} samples")
    noise_x = np.random.normal(0, 0.02, X.shape) 
    noise_y = np.random.normal(0, 0.02, y.shape)
    
    X_aug = X + noise_x
    y_aug = y + noise_y
    return np.vstack((X, X_aug)), np.hstack((y, y_aug))

def train_pollution():
    print(f"🌫️ Starting Geography-Aware Pollution Analysis...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data missing at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        target = 'PM2.5'
        
        # 1. Process & Pivot
        if 'pollutant_id' in df.columns:
            df = process_geo_data(df)
            chemical_features = ['PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']
            # Only keep chemicals that exist in the file
            chemical_features = [c for c in chemical_features if c in df.columns]
        else:
            print("   Using Standard Format.")
            chemical_features = [c for c in df.columns if c != target and df[c].dtype in [float, int]]

        # 2. Drop rows where Target is missing
        df_clean = df.dropna(subset=[target]).copy()
        
        # 3. Impute Missing Chemicals (KNN)
        imputer = KNNImputer(n_neighbors=5)
        df_clean[chemical_features] = imputer.fit_transform(df_clean[chemical_features])
        
        # 4. ADD GEOGRAPHY (The Fix)
        # We convert 'state' into numbers (One-Hot Encoding)
        # e.g., State_Delhi=1, State_Karnataka=0
        if 'state' in df_clean.columns:
            print(f"   🗺️ Encoding Geography ({df_clean['state'].nunique()} states)...")
            df_final = pd.get_dummies(df_clean, columns=['state'], drop_first=True)
        else:
            df_final = df_clean
            
        # Define Features (Chemicals + New State Columns)
        features = [c for c in df_final.columns if c not in ['city', 'station', target]]
        
        print(f"   Training on {len(df_final)} stations.")
        print(f"   Total Features: {len(features)} (Chemistry + Geography)")
        
        X = df_final[features].values.astype(float) # Ensure all float for XGB
        y = df_final[target].values
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # 6. Augment (Trick: We augment everything slightly. Trees handle noisy binaries fine)
    X_train, y_train = augment_continuous(X_train, y_train)
    
    # 7. Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,     # Slower learning for better generalization
        max_depth=7,            # Deeper trees to capture State logic
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("   Training Model...")
    model.fit(X_train, y_train)
    
    # 8. Evaluate
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    mean_val = np.mean(y_test)
    accuracy_pct = max(0, 1.0 - (mae / mean_val)) * 100

    # 9. Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    
    save_metrics(
        model_name="Pollution_XGB_Geo",
        dataset_name="Yash Dogra AQI (Geo-Aware)",
        metrics={
            "RMSE": float(round(rmse, 2)),
            "Accuracy_Pct": float(round(accuracy_pct, 2)),
            "R2_Score": float(round(r2, 4))
        },
        params={"type": "XGBoost + Geography", "features": len(features)},
        train_len=len(X_train),
        test_len=len(X_test)
    )
    print(f"✅ Pollution Geo-Model Saved. Accuracy: {accuracy_pct:.2f}%")

if __name__ == "__main__":
    train_pollution()