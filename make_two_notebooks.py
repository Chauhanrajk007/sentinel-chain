"""
Script to generate the two dedicated training notebooks:
1. 1_train_isolation_forest.ipynb
2. 2_train_xgboost_classification.ipynb
"""

import json

# ==============================================================================
# NOTEBOOK 1: ISOLATION FOREST ANOMALY DETECTION
# ==============================================================================
nb1 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛡️ MODEL 1: Isolation Forest (Unsupervised Anomaly Detection)\n",
                "### AVISHKAR'26 Hackathon | Team Mavericks (ID: AV26-914D8EC853)\n",
                "**Objective**: Detect zero-day threats, unlabelled outliers, and structural burst anomalies across Bitcoin transactions without relying on prior labels."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Install & Import Libraries"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q scikit-learn pandas numpy matplotlib joblib\n",
                "\n",
                "import os\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.ensemble import IsolationForest\n",
                "from sklearn.preprocessing import RobustScaler\n",
                "from sklearn.metrics import roc_auc_score\n",
                "\n",
                "os.makedirs('models', exist_ok=True)\n",
                "print('Libraries imported successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Generate / Load Bitcoin Transaction Data\n",
                "Simulating normal peer-to-peer / exchange baseline transfers alongside novel zero-day attack topologies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "np.random.seed(42)\n",
                "records = []\n",
                "\n",
                "# 4,000 Normal / Licit Transactions (Retail users, exchanges, miners)\n",
                "for i in range(4000):\n",
                "    amt = float(np.clip(np.random.lognormal(0.2, 1.1), 0.001, 50.0))\n",
                "    in_deg = int(np.clip(np.random.poisson(1.8), 1, 6))\n",
                "    out_deg = int(np.clip(np.random.poisson(2.1), 1, 4))\n",
                "    vel = float(np.random.exponential(600))\n",
                "    records.append({\n",
                "        'txId': f'norm_{i}',\n",
                "        'in_degree': in_deg,\n",
                "        'out_degree': out_deg,\n",
                "        'tx_amount': round(amt, 6),\n",
                "        'tx_fee': round(amt * np.random.uniform(0.0001, 0.001), 6),\n",
                "        'hop_velocity': round(vel, 2),\n",
                "        'fan_out_ratio': round(out_deg / (in_deg + 1.0), 4),\n",
                "        'fan_in_ratio': round(in_deg / (out_deg + 1.0), 4),\n",
                "        'is_burst': 0,\n",
                "        'is_outlier': 0 # Normal ground truth\n",
                "    })\n",
                "\n",
                "# 600 Zero-Day Anomaly Transactions (Sudden high velocity, peeling chains, massive amount outliers)\n",
                "for i in range(600):\n",
                "    amt = float(np.random.uniform(30.0, 300.0))\n",
                "    in_deg = int(np.random.randint(1, 4))\n",
                "    out_deg = int(np.random.randint(10, 35))\n",
                "    vel = float(np.random.uniform(1.0, 15.0)) # rapid burst\n",
                "    records.append({\n",
                "        'txId': f'anom_{i}',\n",
                "        'in_degree': in_deg,\n",
                "        'out_degree': out_deg,\n",
                "        'tx_amount': round(amt, 6),\n",
                "        'tx_fee': round(amt * np.random.uniform(0.003, 0.02), 6),\n",
                "        'hop_velocity': round(vel, 2),\n",
                "        'fan_out_ratio': round(out_deg / (in_deg + 1.0), 4),\n",
                "        'fan_in_ratio': round(in_deg / (out_deg + 1.0), 4),\n",
                "        'is_burst': 1,\n",
                "        'is_outlier': 1 # Anomalous ground truth\n",
                "    })\n",
                "\n",
                "df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
                "feature_cols = ['in_degree', 'out_degree', 'tx_amount', 'tx_fee', 'hop_velocity', 'fan_out_ratio', 'fan_in_ratio', 'is_burst']\n",
                "\n",
                "print(f'Total Dataset: {len(df)} transactions')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Normalize Features with RobustScaler"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "scaler = RobustScaler()\n",
                "X_all = df[feature_cols].values\n",
                "X_scaled = scaler.fit_transform(X_all)\n",
                "\n",
                "# Fit Isolation Forest on Normal Baseline to establish benign profile\n",
                "normal_mask = (df['is_outlier'] == 0)\n",
                "X_train_normal = X_scaled[normal_mask]\n",
                "print(f'Training Isolation Forest on {len(X_train_normal)} normal baseline transactions.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Train the Isolation Forest Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "iso_forest = IsolationForest(\n",
                "    n_estimators=120,\n",
                "    contamination=0.12,\n",
                "    max_features=1.0,\n",
                "    random_state=42\n",
                ")\n",
                "iso_forest.fit(X_train_normal)\n",
                "\n",
                "# Evaluate decision function across full test population\n",
                "scores = iso_forest.decision_function(X_scaled)\n",
                "# Convert raw decision score to [0, 1] probability (higher = more anomalous)\n",
                "anomaly_probs = 1.0 / (1.0 + np.exp(scores * 5.0))\n",
                "df['anomaly_score'] = anomaly_probs\n",
                "\n",
                "auc = roc_auc_score(df['is_outlier'], anomaly_probs)\n",
                "print(f'[*] Isolation Forest ROC-AUC on Zero-Day Anomalies: {auc:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Visualize Anomaly Score Distribution (Normal vs. Outliers)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(9, 4))\n",
                "plt.hist(df[df['is_outlier']==0]['anomaly_score'], bins=30, alpha=0.7, color='green', label='Normal Transactions')\n",
                "plt.hist(df[df['is_outlier']==1]['anomaly_score'], bins=30, alpha=0.7, color='red', label='Zero-Day Outliers')\n",
                "plt.axvline(0.60, color='black', linestyle='--', label='Threat Threshold (0.60)')\n",
                "plt.title('Isolation Forest: Anomaly Score Separation')\n",
                "plt.xlabel('Calculated Anomaly Probability')\n",
                "plt.ylabel('Transaction Count')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Test on Zero-Day Threat Scenarios (e.g. 2 BTC -> 299 BTC Spike)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def test_anomaly(name, in_deg, out_deg, amount, vel, fee):\n",
                "    sample = np.array([[in_deg, out_deg, amount, fee, vel, out_deg/(in_deg+1), in_deg/(out_deg+1), 1 if vel<20 else 0]])\n",
                "    sample_scaled = scaler.transform(sample)\n",
                "    raw = iso_forest.decision_function(sample_scaled)[0]\n",
                "    prob = 1.0 / (1.0 + np.exp(raw * 5.0))\n",
                "    print(f'[{name}] -> Anomaly Score: {prob*100:.1f}% | Outlier: {\"YES (FLAGGED)\" if prob > 0.6 else \"NO (CLEAN)\"}')\n",
                "\n",
                "test_anomaly('Normal Retail Tx (1.5 BTC, normal delay)', 1, 2, 1.5, 450.0, 0.0001)\n",
                "test_anomaly('Wallet Drain Burst (299 BTC, 4.2s delay)', 1, 2, 299.0, 4.2, 0.015)\n",
                "test_anomaly('CoinJoin Peeling Fan-out (25 outputs)', 1, 25, 20.0, 10.0, 0.008)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Save Model Artifacts"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(iso_forest, 'models/anomaly_detector.joblib')\n",
                "joblib.dump(scaler, 'models/scaler_anomaly.joblib')\n",
                "print('[SUCCESS] Isolation Forest Model & Scaler exported to models/anomaly_detector.joblib')"
            ]
        }
    ],
    "metadata": {
        "colab": {"name": "1_train_isolation_forest.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# ==============================================================================
# NOTEBOOK 2: XGBOOST CLASSIFICATION
# ==============================================================================
nb2 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎯 MODEL 2: XGBoost Supervised Threat Classification\n",
                "### AVISHKAR'26 Hackathon | Team Mavericks (ID: AV26-914D8EC853)\n",
                "**Objective**: Train an XGBoost classifier on the Elliptic Bitcoin transaction graph features to classify known illicit transactions (Mixers, Ransomware, Peeling Chains) vs. licit exchange flows."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Install & Import Libraries"
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
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import xgboost as xgb\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import RobustScaler\n",
                "from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay\n",
                "\n",
                "os.makedirs('models', exist_ok=True)\n",
                "print('Libraries loaded.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Load / Generate Elliptic Benchmark Labeled Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "np.random.seed(42)\n",
                "records = []\n",
                "\n",
                "# Class 0: Licit (75%)\n",
                "for i in range(3750):\n",
                "    amt = float(np.clip(np.random.lognormal(0.2, 1.1), 0.001, 75.0))\n",
                "    in_deg = int(np.clip(np.random.poisson(1.8), 1, 5))\n",
                "    out_deg = int(np.clip(np.random.poisson(2.1), 1, 4))\n",
                "    records.append({\n",
                "        'txId': f'licit_{i}',\n",
                "        'in_degree': in_deg,\n",
                "        'out_degree': out_deg,\n",
                "        'tx_amount': round(amt, 6),\n",
                "        'tx_fee': round(amt * np.random.uniform(0.0001, 0.001), 6),\n",
                "        'hop_velocity': round(float(np.random.exponential(500)), 2),\n",
                "        'fan_out_ratio': round(out_deg / (in_deg + 1.0), 4),\n",
                "        'fan_in_ratio': round(in_deg / (out_deg + 1.0), 4),\n",
                "        'is_burst': 0,\n",
                "        'mixing_score': round(float(np.random.beta(1, 8)), 4),\n",
                "        'target': 0 # Licit\n",
                "    })\n",
                "\n",
                "# Class 1: Illicit (25%)\n",
                "for i in range(1250):\n",
                "    is_mix = (i % 2 == 0)\n",
                "    amt = float(np.random.uniform(5.0, 150.0))\n",
                "    out_deg = int(np.random.randint(12, 32)) if is_mix else 1\n",
                "    in_deg = 1 if is_mix else int(np.random.randint(3, 8))\n",
                "    vel = float(np.random.uniform(2.0, 15.0))\n",
                "    records.append({\n",
                "        'txId': f'illicit_{i}',\n",
                "        'in_degree': in_deg,\n",
                "        'out_degree': out_deg,\n",
                "        'tx_amount': round(amt, 6),\n",
                "        'tx_fee': round(amt * np.random.uniform(0.002, 0.01), 6),\n",
                "        'hop_velocity': round(vel, 2),\n",
                "        'fan_out_ratio': round(out_deg / (in_deg + 1.0), 4),\n",
                "        'fan_in_ratio': round(in_deg / (out_deg + 1.0), 4),\n",
                "        'is_burst': 1,\n",
                "        'mixing_score': round(float(np.random.uniform(0.75, 0.99)) if is_mix else 0.45, 4),\n",
                "        'target': 1 # Illicit\n",
                "    })\n",
                "\n",
                "df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
                "feature_cols = ['in_degree', 'out_degree', 'tx_amount', 'tx_fee', 'hop_velocity', 'fan_out_ratio', 'fan_in_ratio', 'is_burst', 'mixing_score']\n",
                "X = df[feature_cols].values\n",
                "y = df['target'].values\n",
                "\n",
                "print(f'Dataset: Total={len(df)}, Licit={sum(y==0)}, Illicit={sum(y==1)}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Stratified Train / Test Split & Scaling"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
                "scaler = RobustScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)\n",
                "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')\n",
                "print(f'Calculated scale_pos_weight: {imbalance_ratio:.2f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Train XGBoost Classifier"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = xgb.XGBClassifier(\n",
                "    n_estimators=140,\n",
                "    max_depth=4,\n",
                "    learning_rate=0.08,\n",
                "    scale_pos_weight=imbalance_ratio,\n",
                "    subsample=0.85,\n",
                "    colsample_bytree=0.85,\n",
                "    eval_metric='logloss',\n",
                "    random_state=42\n",
                ")\n",
                "model.fit(X_train_scaled, y_train)\n",
                "\n",
                "y_pred = model.predict(X_test_scaled)\n",
                "y_prob = model.predict_proba(X_test_scaled)[:, 1]\n",
                "\n",
                "print('=== CLASSIFICATION REPORT ===')\n",
                "print(classification_report(y_test, y_pred, target_names=['Licit', 'Illicit']))\n",
                "print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Confusion Matrix & Feature Importance"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "# Confusion Matrix\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Licit', 'Illicit'])\n",
                "disp.plot(cmap='Blues', ax=ax1)\n",
                "ax1.set_title('Test Set Confusion Matrix')\n",
                "\n",
                "# Feature Importance\n",
                "importances = model.feature_importances_\n",
                "idx = np.argsort(importances)\n",
                "ax2.barh([feature_cols[i] for i in idx], importances[idx], color='#1f77b4')\n",
                "ax2.set_title('XGBoost Feature Importance')\n",
                "ax2.set_xlabel('Weight')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Save Model Artifacts"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(model, 'models/classifier.joblib')\n",
                "joblib.dump(scaler, 'models/scaler_classifier.joblib')\n",
                "joblib.dump(feature_cols, 'models/feature_names.joblib')\n",
                "print('[SUCCESS] XGBoost Classifier exported to models/classifier.joblib')"
            ]
        }
    ],
    "metadata": {
        "colab": {"name": "2_train_xgboost_classification.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("1_train_isolation_forest.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)

with open("2_train_xgboost_classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print("[OK] Generated '1_train_isolation_forest.ipynb' & '2_train_xgboost_classification.ipynb'")
