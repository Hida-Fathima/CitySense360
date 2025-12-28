import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import sys
import joblib

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.metrics_logger import save_metrics

# --- CONFIG ---
DATA_PATH = "data/complaints/complaints.json"
MODEL_PATH = "models/complaints_bert/"
LABEL_ENCODER_PATH = "models/complaints_label_encoder.pkl"

EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 128        # Real complaints are longer
MIN_SAMPLES = 100    # Keep depts with >100 complaints (adjust based on file size)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    print("📂 Loading Real Complaint JSON...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data missing at {DATA_PATH}")
        return None

    try:
        # Load JSON (Handle list of objects)
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        # 'subject_content_text' (Input) and 'org_code' (Label)
        text_col = 'subject_content_text'
        label_col = 'org_code'
        
        if text_col not in df.columns or label_col not in df.columns:
            print(f"❌ Columns missing. Available: {df.columns}")
            return None
            
        # Filter: Drop missing
        df = df.dropna(subset=[text_col, label_col])
        
        # SMART FILTER: Keep Major Departments
        dept_counts = df[label_col].value_counts()
        valid_depts = dept_counts[dept_counts > MIN_SAMPLES].index.tolist()
        
        print(f"   📉 Filter: Keeping {len(valid_depts)} Major Depts (out of {len(dept_counts)})")
        
        df = df[df[label_col].isin(valid_depts)].copy()
        
        # Clean Text
        df['text_clean'] = df[text_col].astype(str).str.slice(0, 512) # Truncate massive texts
        
        print(f"   ✓ Training on {len(df)} Real Complaints.")
        print(f"   Example: {df['text_clean'].iloc[0][:50]}... -> {df[label_col].iloc[0]}")
        
        return df, label_col

    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_complaints():
    print(f"🗣️ Starting JSON-Based BERT Training on {DEVICE}...")

    result = load_data()
    if not result: return
    df, label_col = result

    # Encode Labels
    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df[label_col])
    num_classes = len(label_encoder.classes_)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label_id'], test_size=0.2, stratify=df['label_id'], random_state=42
    )

    # BERT Setup
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    model = model.to(DEVICE)

    train_ds = ComplaintDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer, MAX_LEN)
    test_ds = ComplaintDataset(X_test.to_numpy(), y_test.to_numpy(), tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train
    for epoch in range(EPOCHS):
        print(f"   Epoch {epoch+1}/{EPOCHS}")
        model.train()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0 and step > 0:
                print(f"     Step {step}: Loss {loss.item():.4f}")

    # Evaluate
    print("   Evaluating...")
    model.eval()
    preds, true_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=mask)
            _, prediction = torch.max(outputs.logits, dim=1)
            
            preds.extend(prediction.cpu().tolist())
            true_labels.extend(batch['labels'].cpu().tolist())

    acc = accuracy_score(true_labels, preds)
    
    # Save
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    save_metrics(
        model_name="Complaint_BERT",
        dataset_name=f"Real Complaints ({num_classes} Depts)",
        metrics={"Accuracy_Pct": float(round(acc * 100, 2))},
        params={"type": "DistilBERT (Real Data)", "classes": num_classes},
        train_len=len(X_train),
        test_len=len(X_test)
    )
    print(f"✅ Complaint Model Saved. Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_complaints()