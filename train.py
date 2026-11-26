"""
Cryptocurrency Transaction Threat Detection - Dual Model Training Pipeline
Trains:
1. Supervised XGBoost Classifier (Licit vs Illicit)
2. Unsupervised Isolation Forest Anomaly Detector (Outlier / Zero-day bursts)
3. Computes explainability weights and saves production artifacts
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
from data_generator import generate_synthetic_elliptic_dataset

def load_or_create_data(data_dir="data"):
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    
    if not os.path.exists(feat_path) or not os.path.exists(class_path):
        print("[!] Elliptic dataset not found locally. Generating synthetic Elliptic-structured dataset...")
        feat_path, class_path = generate_synthetic_elliptic_dataset(output_dir=data_dir)
    else:
        print(f"[*] Found dataset: '{feat_path}' and '{class_path}'")
        
    features_df = pd.read_csv(feat_path)
    classes_df = pd.read_csv(class_path)
    
    # Merge on txId
    # If the file is official Elliptic without headers:
    if "txId" not in features_df.columns:
        features_df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(len(features_df.columns) - 2)]
    if "txId" not in classes_df.columns:
        classes_df.columns = ["txId", "class"]
        
    df = pd.merge(features_df, classes_df, on="txId")
    return df

def train_dual_models(data_dir="data", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    df = load_or_create_data(data_dir)
    
    # Identify feature columns
    ignore_cols = ["txId", "class", "time_step"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Separate labeled vs unknown
    labeled_df = df[df["class"].astype(str).isin(["1", "2", 1, 2])].copy()
    labeled_df["target"] = labeled_df["class"].apply(lambda x: 1 if str(x) == "1" else 0)
    
    print(f"[*] Dataset summary: Total={len(df)}, Labeled={len(labeled_df)}, Illicit={sum(labeled_df['target']==1)}, Licit={sum(labeled_df['target']==0)}")
    
    X = labeled_df[feature_cols].values
    y = labeled_df["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # MODEL 1: Supervised XGBoost Classifier
    # -------------------------------------------------------------
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    print(f"[*] Training XGBoost Classifier with scale_pos_weight={pos_weight:.2f}...")
    
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"[XGBoost Metrics] Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # -------------------------------------------------------------
    # MODEL 2: Unsupervised Anomaly Detector (Isolation Forest)
    # -------------------------------------------------------------
    # Train on normal licit samples + background to learn baseline topology
    print("[*] Training Isolation Forest Anomaly Detector...")
    normal_indices = np.where(y_train == 0)[0]
    X_train_normal = X_train_scaled[normal_indices]
    
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        max_features=0.9,
        random_state=42
    )
    anomaly_detector.fit(X_train_normal)
    
    # Anomaly scores on test set
    anomaly_raw_scores = anomaly_detector.decision_function(X_test_scaled)
    # Map decision function (-0.5 to 0.5 typical) into [0, 1] anomaly score (higher = more anomalous)
    anomaly_probs = 1.0 / (1.0 + np.exp(anomaly_raw_scores * 5.0))
    
    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [{"feature": feature_cols[i], "importance": float(importances[i])} for i in sorted_idx[:10]]
    
    # Save artifacts
    metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "total_samples": len(df),
        "labeled_samples": len(labeled_df),
        "illicit_count": int(sum(labeled_df['target']==1)),
        "licit_count": int(sum(labeled_df['target']==0)),
        "top_features": top_features,
        "feature_cols": feature_cols
    }
    
    joblib.dump(clf, os.path.join(models_dir, "classifier.joblib"))
    joblib.dump(anomaly_detector, os.path.join(models_dir, "anomaly_detector.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_names.joblib"))
    joblib.dump(metrics, os.path.join(models_dir, "metrics.joblib"))
    
    print(f"[OK] Both models trained and saved to '{models_dir}/'")
    return metrics

if __name__ == "__main__":
    train_dual_models()
