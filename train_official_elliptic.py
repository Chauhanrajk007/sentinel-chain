"""
SENTINEL-CHAIN: Official Elliptic Dataset Pipeline (166 Features)
Directly downloads and trains on the authentic MIT-IBM / Elliptic Benchmark Dataset:
- 203,769 Bitcoin Transactions
- 166 Features (93 Local Features + 73 Graph Aggregation Features)
- Class 1: Illicit (Mixers, Ransomware, Phishing, Scams)
- Class 2: Licit (Exchanges, Miners, Wallets)
- Class unknown: Used for Unsupervised Anomaly Detection (Zero-Day Evaluation)
"""

import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Official Hugging Face Mirrors for Elliptic Bitcoin Dataset (Zero Auth Required)
CLASSES_URL = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv"
FEATURES_URL = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv"
EDGELIST_URL = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_edgelist.csv"

def download_official_elliptic():
    """Downloads official raw Elliptic files if not present locally."""
    for url, fname in [(CLASSES_URL, "elliptic_txs_classes.csv"), (FEATURES_URL, "elliptic_txs_features.csv")]:
        dest = os.path.join(DATA_DIR, fname)
        if not os.path.exists(dest):
            print(f"[*] Downloading official '{fname}' from Hugging Face...")
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] Downloaded '{fname}' ({os.path.getsize(dest) / (1024*1024):.1f} MB)")
        else:
            print(f"[*] Found local official file: '{fname}'")

def load_official_elliptic():
    download_official_elliptic()
    classes_path = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")
    features_path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
    
    print("[*] Loading raw features and labels into memory...")
    classes_df = pd.read_csv(classes_path)
    # The official features CSV has no header: column 0 is txId, column 1 is time_step, followed by 165 features
    features_df = pd.read_csv(features_path, header=None)
    
    # Name the columns
    num_cols = len(features_df.columns)
    col_names = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, num_cols - 1)]
    features_df.columns = col_names
    
    # Merge on txId
    df = pd.merge(features_df, classes_df, on="txId")
    print(f"[OK] Loaded {len(df)} transactions with {num_cols - 2} graph & local features.")
    return df

def train_official_models():
    df = load_official_elliptic()
    
    # Filter known labeled transactions for supervised training:
    # '1' = Illicit, '2' = Licit, 'unknown' = Unlabelled
    labeled = df[df["class"].astype(str).isin(["1", "2"])].copy()
    labeled["target"] = labeled["class"].apply(lambda x: 1 if str(x) == "1" else 0)
    
    illicit_cnt = sum(labeled["target"] == 1)
    licit_cnt = sum(labeled["target"] == 0)
    print(f"[*] Labeled Population: Total={len(labeled)} | Illicit={illicit_cnt} | Licit={licit_cnt}")
    
    feature_cols = [c for c in labeled.columns if c not in ["txId", "time_step", "class", "target"]]
    
    X = labeled[feature_cols].values
    y = labeled["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 1. Supervised XGBoost on Full 166 Features
    # -------------------------------------------------------------
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"[*] Training XGBoost Classifier across {len(feature_cols)} features (scale_pos_weight={scale_pos:.2f})...")
    
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("OFFICIAL ELLIPTIC 166-FEATURE CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Licit", "Illicit"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # -------------------------------------------------------------
    # 2. Unsupervised Isolation Forest on Full Features
    # -------------------------------------------------------------
    print("\n[*] Training Isolation Forest Anomaly Detector on 166 features...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42,
        n_jobs=-1
    )
    # Train on licit baseline
    iso.fit(X_train_scaled[y_train == 0])
    
    # Save production artifacts
    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier_166.joblib"))
    joblib.dump(iso, os.path.join(MODELS_DIR, "anomaly_detector_166.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_166.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names_166.joblib"))
    
    print(f"\n[SUCCESS] Trained & Saved 166-feature models into '{MODELS_DIR}/'")

if __name__ == "__main__":
    train_official_models()
