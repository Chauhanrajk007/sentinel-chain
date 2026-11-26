"""
Generates the Colab Notebook for the Official 166-Feature Elliptic Training.
"""
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏛️ OFFICIAL MIT-IBM / ELLIPTIC BENCHMARK (166 FEATURES)\n",
                "### AVISHKAR'26 Hackathon | Team Mavericks (ID: AV26-914D8EC853)\n",
                "**Dataset**: Official Elliptic Bitcoin Dataset (203,769 transactions across 49 time-steps)\n",
                "**Feature Architecture**: Exactly **166 Features** (93 Local Transaction Features + 73 Graph Neighborhood Aggregation Features)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Install Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q xgboost scikit-learn pandas numpy matplotlib joblib\n",
                "\n",
                "import os\n",
                "import urllib.request\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import xgboost as xgb\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import RobustScaler\n",
                "from sklearn.ensemble import IsolationForest\n",
                "from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay\n",
                "\n",
                "os.makedirs('data', exist_ok=True)\n",
                "os.makedirs('models', exist_ok=True)\n",
                "print('Environment initialized.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Download Official Elliptic Benchmark CSVs (Direct from Hugging Face Mirror)\n",
                "No API key, no account needed."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!wget -O data/elliptic_txs_classes.csv https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv\n",
                "!wget -O data/elliptic_txs_features.csv https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv\n",
                "\n",
                "!ls -lh data/"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Load & Structure the 166 Official Features\n",
                "The original Elliptic dataset paper by Weber et al. defines:\n",
                "- **Features 1 to 93**: Local transaction features (fee, output BTC, degree, velocity, variance, skewness)\n",
                "- **Features 94 to 166 (73 features)**: 1-hop backward/forward graph neighbor aggregated features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Loading official features...')\n",
                "classes_df = pd.read_csv('data/elliptic_txs_classes.csv')\n",
                "features_df = pd.read_csv('data/elliptic_txs_features.csv', header=None)\n",
                "\n",
                "# Name the 166 features\n",
                "num_features = len(features_df.columns) - 2\n",
                "features_df.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, num_features + 1)]\n",
                "\n",
                "# Merge with labels\n",
                "df = pd.merge(features_df, classes_df, on='txId')\n",
                "print(f'[OK] Loaded {len(df):,} transactions with exactly {num_features} features!')\n",
                "\n",
                "# Filter labeled for supervised training: 1 = Illicit, 2 = Licit\n",
                "labeled = df[df['class'].astype(str).isin(['1', '2'])].copy()\n",
                "labeled['target'] = labeled['class'].apply(lambda x: 1 if str(x) == '1' else 0)\n",
                "\n",
                "print(f'Labeled Transactions: {len(labeled):,} (Illicit={sum(labeled[\"target\"]==1):,}, Licit={sum(labeled[\"target\"]==0):,})')\n",
                "print(f'Unlabelled Transactions (for Anomaly Detection): {sum(df[\"class\"]==\"unknown\"):,}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Train/Test Split (80/20 Stratified)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [f'feat_{i}' for i in range(1, num_features + 1)]\n",
                "X = labeled[feature_cols].values\n",
                "y = labeled['target'].values\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
                "\n",
                "scaler = RobustScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)\n",
                "print(f'Training matrix: {X_train.shape} | Test matrix: {X_test.shape}')\n",
                "print(f'Imbalance scale_pos_weight: {pos_weight:.2f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Train Supervised XGBoost Classifier on 166 Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Training XGBoost Classifier across all 166 features...')\n",
                "clf = xgb.XGBClassifier(\n",
                "    n_estimators=150,\n",
                "    max_depth=5,\n",
                "    learning_rate=0.08,\n",
                "    scale_pos_weight=pos_weight,\n",
                "    subsample=0.85,\n",
                "    colsample_bytree=0.85,\n",
                "    eval_metric='logloss',\n",
                "    random_state=42,\n",
                "    n_jobs=-1\n",
                ")\n",
                "clf.fit(X_train_scaled, y_train)\n",
                "\n",
                "y_pred = clf.predict(X_test_scaled)\n",
                "y_prob = clf.predict_proba(X_test_scaled)[:, 1]\n",
                "\n",
                "print('='*60)\n",
                "print('OFFICIAL 166-FEATURE CLASSIFICATION REPORT')\n",
                "print('='*60)\n",
                "print(classification_report(y_test, y_pred, target_names=['Licit', 'Illicit']))\n",
                "print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Train Isolation Forest Anomaly Detector on 166 Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Training Isolation Forest Anomaly Detector...')\n",
                "iso = IsolationForest(n_estimators=100, contamination=0.10, random_state=42, n_jobs=-1)\n",
                "iso.fit(X_train_scaled[y_train == 0])\n",
                "\n",
                "scores = iso.decision_function(X_test_scaled)\n",
                "anomaly_probs = 1.0 / (1.0 + np.exp(scores * 5.0))\n",
                "\n",
                "print(f'Isolation Forest fitted successfully on {sum(y_train==0):,} licit baseline transactions.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Feature Importance Analysis (Top Features Driving Detection)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importances = clf.feature_importances_\n",
                "top_indices = np.argsort(importances)[::-1][:15]\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.barh([f'feat_{i+1}' for i in top_indices][::-1], importances[top_indices][::-1], color='#2563eb')\n",
                "plt.title('Top 15 Most Discriminative Features (Official Elliptic 166 Features)')\n",
                "plt.xlabel('XGBoost Relative Feature Importance')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Export Production Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(clf, 'models/classifier.joblib')\n",
                "joblib.dump(iso, 'models/anomaly_detector.joblib')\n",
                "joblib.dump(scaler, 'models/scaler.joblib')\n",
                "print('[SUCCESS] Trained models saved. Ready to connect to the Streamlit Dashboard and REST API!')"
            ]
        }
    ],
    "metadata": {
        "colab": {"name": "train_official_elliptic_166_features.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("train_official_elliptic_166_features.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[OK] Generated 'train_official_elliptic_166_features.ipynb'")
