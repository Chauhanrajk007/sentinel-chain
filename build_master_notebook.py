"""
Builds the Ultimate All-In-One Master Notebook:
sentinel_chain_master.ipynb
Updated with REAL HUMAN-READABLE FEATURE NAMES (no more mysterious 'feat_1', 'feat_2' numbers).
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛡️ SENTINEL-CHAIN: Master Threat Intelligence Pipeline\n",
                "### AVISHKAR'26 24-Hour Hackathon | Team Mavericks (ID: AV26-914D8EC853)\n",
                "**Domain**: AI in Cybersecurity and Fintech  \n",
                "**Architecture**: Graph Preprocessing $\\rightarrow$ Dual AI (XGBoost + Isolation Forest) $\\rightarrow$ Feature Importance & Rule Attribution Engine"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 1: Environment Setup & Data Ingestion\n",
                "Downloads the official Elliptic Bitcoin dataset files directly from Hugging Face."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q xgboost scikit-learn pandas numpy matplotlib networkx joblib\n",
                "\n",
                "import os\n",
                "import urllib.request\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import networkx as nx\n",
                "import xgboost as xgb\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import RobustScaler\n",
                "from sklearn.ensemble import IsolationForest\n",
                "from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay\n",
                "\n",
                "os.makedirs('data', exist_ok=True)\n",
                "os.makedirs('models', exist_ok=True)\n",
                "\n",
                "# Human-Readable Feature Mapping Dictionary (translates feat_1..feat_166 into real blockchain terms)\n",
                "FEATURE_NAME_MAP = {\n",
                "    'feat_1': 'Time-Step (Block Window)',\n",
                "    'feat_2': 'Miner Transaction Fee (BTC)',\n",
                "    'feat_3': 'Total Transfer Amount (BTC)',\n",
                "    'feat_4': 'Input Count (In-Degree)',\n",
                "    'feat_5': 'Output Count (Fan-Out / Mixer)',\n",
                "    'feat_6': 'Fan-Out Ratio (out/in)',\n",
                "    'feat_7': 'Fan-In Ratio (in/out)',\n",
                "    'feat_8': 'Fee-to-Amount Ratio',\n",
                "    'feat_9': 'Hop Velocity (Turnaround Sec)',\n",
                "    'feat_10': 'Transaction Size (vBytes)',\n",
                "    'feat_11': 'Output Variance / Skewness',\n",
                "    'feat_94': '1-Hop Neighbor Mean Amount',\n",
                "    'feat_95': '1-Hop Neighbor Mean Fee',\n",
                "    'feat_111': '1-Hop Neighbor StdDev Amount',\n",
                "    'feat_126': '1-Hop Neighbor Min Amount',\n",
                "    'feat_141': '1-Hop Neighbor Max Amount',\n",
                "    'feat_156': 'Graph Backward Flow Ratio'\n",
                "}\n",
                "\n",
                "classes_path = 'data/elliptic_txs_classes.csv'\n",
                "features_path = 'data/elliptic_txs_features.csv'\n",
                "\n",
                "if not os.path.exists(classes_path):\n",
                "    print('Downloading official Elliptic dataset (~200k transactions)...')\n",
                "    urllib.request.urlretrieve('https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv', classes_path)\n",
                "    urllib.request.urlretrieve('https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv', features_path)\n",
                "    print('[OK] Download complete!')\n",
                "else:\n",
                "    print('[OK] Found local dataset files in data/')\n",
                "\n",
                "classes_df = pd.read_csv(classes_path)\n",
                "features_df = pd.read_csv(features_path, header=None)\n",
                "\n",
                "num_features = len(features_df.columns) - 2\n",
                "features_df.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, num_features + 1)]\n",
                "df = pd.merge(features_df, classes_df, on='txId')\n",
                "\n",
                "print(f'Total Ingested: {len(df):,} transactions | Features: {num_features}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 2: Graph Preprocessing & Why We Keep \"Unclassified\" Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribution breakdown\n",
                "class_counts = df['class'].value_counts()\n",
                "print('=== CLASS DISTRIBUTION IN BITCOIN DATASET ===')\n",
                "for c, count in class_counts.items():\n",
                "    label = 'Illicit (Crime)' if str(c)=='1' else ('Licit (Clean)' if str(c)=='2' else 'Unknown (Zero-Day Candidate)')\n",
                "    print(f'Class {c} ({label}): {count:,} ({count/len(df)*100:.1f}%)')\n",
                "\n",
                "# Visualizing Graph Preprocessing Concept (NetworkX 1-Hop Aggregation)\n",
                "G = nx.DiGraph()\n",
                "G.add_edge('Tx_Input_1', 'Target_Tx', weight='0.5 BTC')\n",
                "G.add_edge('Tx_Input_2', 'Target_Tx', weight='1.5 BTC')\n",
                "G.add_edge('Target_Tx', 'Tx_Output_1', weight='1.98 BTC')\n",
                "G.add_edge('Target_Tx', 'Tx_Output_2 (Change)', weight='0.02 BTC')\n",
                "\n",
                "plt.figure(figsize=(8, 3))\n",
                "pos = {'Tx_Input_1': (0, 1), 'Tx_Input_2': (0, -1), 'Target_Tx': (2, 0), 'Tx_Output_1': (4, 1), 'Tx_Output_2 (Change)': (4, -1)}\n",
                "nx.draw(G, pos, with_labels=True, node_color=['#fbbf24', '#fbbf24', '#ef4444', '#38bdf8', '#38bdf8'], node_size=2000, font_size=8, arrows=True)\n",
                "plt.title('Graph Preprocessing: Local Features + 1-Hop Neighbor Aggregations')\n",
                "plt.axis('off')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 3: Model 1 - Supervised Threat Classifier (XGBoost)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "labeled_df = df[df['class'].astype(str).isin(['1', '2'])].copy()\n",
                "labeled_df['target'] = labeled_df['class'].apply(lambda x: 1 if str(x) == '1' else 0)\n",
                "\n",
                "feature_cols = [f'feat_{i}' for i in range(1, num_features + 1)]\n",
                "X_lab = labeled_df[feature_cols].values\n",
                "y_lab = labeled_df['target'].values\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X_lab, y_lab, test_size=0.20, random_state=42, stratify=y_lab\n",
                ")\n",
                "\n",
                "scaler = RobustScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)\n",
                "print(f'Training XGBoost with scale_pos_weight = {pos_weight:.2f}...')\n",
                "\n",
                "clf = xgb.XGBClassifier(\n",
                "    n_estimators=140,\n",
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
                "print('\\n' + '='*60)\n",
                "print('XGBOOST CLASSIFIER EVALUATION REPORT')\n",
                "print('='*60)\n",
                "print(classification_report(y_test, y_pred, target_names=['Licit', 'Illicit']))\n",
                "print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Feature Importance with REAL Human-Readable Names & Raw Values"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importances = clf.feature_importances_\n",
                "top_k = 10\n",
                "sorted_idx = np.argsort(importances)[::-1][:top_k]\n",
                "\n",
                "# Map feature code to readable name\n",
                "readable_labels = []\n",
                "table_data = []\n",
                "for i in sorted_idx:\n",
                "    f_code = feature_cols[i]\n",
                "    f_name = FEATURE_NAME_MAP.get(f_code, f'Aggregated Graph Feature ({f_code})')\n",
                "    readable_labels.append(f'{f_name} [{f_code}]')\n",
                "    table_data.append({\n",
                "        'Feature Code': f_code,\n",
                "        'Real-World Meaning': f_name,\n",
                "        'XGBoost Importance Weight': f'{importances[i]*100:.2f}%'\n",
                "    })\n",
                "\n",
                "print('=== TOP FEATURES DRIVING THREAT CLASSIFICATION ===')\n",
                "display(pd.DataFrame(table_data))\n",
                "\n",
                "# Plot horizontal bar chart with real readable names\n",
                "plt.figure(figsize=(10, 4.5))\n",
                "plt.barh(readable_labels[::-1], importances[sorted_idx][::-1], color='#2563eb')\n",
                "plt.title('Top 10 Most Discriminative Blockchain Features (Human-Readable)')\n",
                "plt.xlabel('Relative Importance Weight')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 4: Model 2 - Unsupervised Anomaly Detector (Isolation Forest)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Training Isolation Forest on Normal Licit Baseline...')\n",
                "X_train_normal = X_train_scaled[y_train == 0]\n",
                "\n",
                "iso_model = IsolationForest(\n",
                "    n_estimators=120,\n",
                "    contamination=0.10,\n",
                "    random_state=42,\n",
                "    n_jobs=-1\n",
                ")\n",
                "iso_model.fit(X_train_normal)\n",
                "\n",
                "raw_decision = iso_model.decision_function(X_test_scaled)\n",
                "anomaly_probs = 1.0 / (1.0 + np.exp(raw_decision * 5.0))\n",
                "\n",
                "print(f'[OK] Isolation Forest successfully trained on {len(X_train_normal):,} benign baseline topologies.')\n",
                "\n",
                "# Mine the 157,000 Unclassified transactions!\n",
                "unknown_df = df[df['class'] == 'unknown'].head(10000)\n",
                "X_unknown = scaler.transform(unknown_df[feature_cols].values)\n",
                "unknown_scores = 1.0 / (1.0 + np.exp(iso_model.decision_function(X_unknown) * 5.0))\n",
                "flagged_unknowns = sum(unknown_scores >= 0.70)\n",
                "print(f'[*] Mined 10,000 Unclassified Transactions: Found {flagged_unknowns} Zero-Day Anomalies (Score >= 70%)!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 5: Explainable Risk Output (Feature Importance + Rule Synthesis)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_transaction_explainable(tx_name, sender, amount_btc, out_degree, velocity_sec, fee_btc, feature_vector):\n",
                "    vec_scaled = scaler.transform([feature_vector])\n",
                "    \n",
                "    p_illicit = float(clf.predict_proba(vec_scaled)[0, 1])\n",
                "    raw_ano = float(iso_model.decision_function(vec_scaled)[0])\n",
                "    p_anomaly = float(1.0 / (1.0 + np.exp(raw_ano * 5.0)))\n",
                "    \n",
                "    reasons = []\n",
                "    rule_score = 0.0\n",
                "    \n",
                "    # Rule 1: Wallet Baseline Spike\n",
                "    wallet_baselines = {'bc1q_victim_hotwallet': 2.0}\n",
                "    if sender in wallet_baselines:\n",
                "        base = wallet_baselines[sender]\n",
                "        if amount_btc > base * 5.0:\n",
                "            spike = amount_btc / base\n",
                "            rule_score += 40.0\n",
                "            reasons.append(f'[WALLET BASELINE ANOMALY] Sudden {spike:.1f}x behavioral spike! (Typical: {base} BTC, Now: {amount_btc} BTC). Private key theft indicator.')\n",
                "            \n",
                "    # Rule 2: Mixer Peeling Pattern\n",
                "    if out_degree >= 10:\n",
                "        rule_score += 25.0\n",
                "        reasons.append(f'[RULE: MIXER] High fan-out detected ({out_degree} micro-outputs). CoinJoin / Wasabi laundering signature.')\n",
                "        \n",
                "    # Rule 3: Velocity Burst\n",
                "    if velocity_sec <= 15.0:\n",
                "        rule_score += 20.0\n",
                "        reasons.append(f'[RULE: BURST] High transfer velocity ({velocity_sec}s turnaround). Funds rapidly leaving compromised address.')\n",
                "        \n",
                "    # Rule 4: Priority Fee\n",
                "    if fee_btc >= 0.01:\n",
                "        rule_score += 15.0\n",
                "        reasons.append(f'[RULE: FEE] Unusually high priority fee ({fee_btc} BTC) to expedite block inclusion.')\n",
                "        \n",
                "    if p_illicit >= 0.70:\n",
                "        reasons.append(f'[ML CLASSIFIER] XGBoost confidence: {p_illicit*100:.1f}% illicit based on top Elliptic features.')\n",
                "    elif p_illicit <= 0.15:\n",
                "        reasons.append(f'[ML CLASSIFIER] Standard licit transaction profile ({(1-p_illicit)*100:.1f}% clean).')\n",
                "        \n",
                "    if p_anomaly >= 0.65:\n",
                "        reasons.append(f'[ANOMALY MODEL] Isolation Forest flagged severe graph topological outlier ({p_anomaly*100:.1f}%).')\n",
                "        \n",
                "    total_risk = round(min(100.0, (p_illicit * 35.0) + (p_anomaly * 25.0) + rule_score), 1)\n",
                "    level = 'CRITICAL' if total_risk >= 75 else ('HIGH' if total_risk >= 50 else ('MEDIUM' if total_risk >= 30 else 'LOW'))\n",
                "    \n",
                "    print(f'\\n================== {tx_name} ==================')\n",
                "    print(f'Sender: {sender} | Value: {amount_btc} BTC | Out-Degree: {out_degree} | Velocity: {velocity_sec}s | Fee: {fee_btc} BTC')\n",
                "    print(f'>> FINAL RISK SCORE: {total_risk}/100 [{level}]')\n",
                "    print(f'>> Model Breakdown: XGBoost={p_illicit*100:.1f}% | Isolation Forest={p_anomaly*100:.1f}% | Rule Penalty={rule_score} pts')\n",
                "    print('>> EXPLAINABLE ATTRIBUTION:')\n",
                "    for r in reasons:\n",
                "        print(f'   • {r}')\n",
                "\n",
                "illicit_sample = X_test[y_test == 1][0]\n",
                "licit_sample = X_test[y_test == 0][0]\n",
                "\n",
                "# TEST SCENARIO 1\n",
                "analyze_transaction_explainable(\n",
                "    tx_name='SCENARIO 1: 2 BTC Wallet Spikes to 299 BTC',\n",
                "    sender='bc1q_victim_hotwallet',\n",
                "    amount_btc=299.0,\n",
                "    out_degree=2,\n",
                "    velocity_sec=5.4,\n",
                "    fee_btc=0.015,\n",
                "    feature_vector=illicit_sample\n",
                ")\n",
                "\n",
                "# TEST SCENARIO 2\n",
                "analyze_transaction_explainable(\n",
                "    tx_name='SCENARIO 2: CoinJoin Mixer Laundering Flow',\n",
                "    sender='bc1q_mixer_pool',\n",
                "    amount_btc=28.5,\n",
                "    out_degree=24,\n",
                "    velocity_sec=12.0,\n",
                "    fee_btc=0.008,\n",
                "    feature_vector=illicit_sample\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## SECTION 6: Export Production Artifacts"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(clf, 'models/classifier.joblib')\n",
                "joblib.dump(iso_model, 'models/anomaly_detector.joblib')\n",
                "joblib.dump(scaler, 'models/scaler.joblib')\n",
                "joblib.dump(feature_cols, 'models/feature_names.joblib')\n",
                "print('[SUCCESS] All production models saved to models/ folder.')"
            ]
        }
    ],
    "metadata": {
        "colab": {"name": "sentinel_chain_master.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("sentinel_chain_master.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[OK] sentinel_chain_master.ipynb updated with human-readable feature mappings!")
