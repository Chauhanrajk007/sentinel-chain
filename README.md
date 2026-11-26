# SENTINEL-CHAIN: Real-Time Crypto Transaction Threat Detection & Graph Reasoning

> **AVISHKAR'26 24-Hour Hackathon**  
> **Domain**: AI in Cybersecurity and Fintech  
> **Problem Statement #30**: Cryptocurrency Transaction Threat Detection  
> **Team ID**: `AV26-914D8EC853` | **Team Name**: `Mavericks`

---

## 🎯 Solution Architecture Overview

```
                      [ LIVE INGESTION LAYER ]
          ┌──────────────────────────────────────────────┐
          │   • Public Bitcoin Testnet API (Mempool)     │
          │   • Elliptic Benchmark Synthetic Streamer    │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
                     [ DUAL-MODEL ML LAYER ]
          ┌──────────────────────────────────────────────┐
          │ 1. XGBoost Classifier (P_illicit)            │
          │ 2. Isolation Forest Anomaly (P_anomaly)      │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
                     [ GRAPH & RULE REASONING ]
          ┌──────────────────────────────────────────────┐
          │ • NetworkX Dynamic DAG & Peeling Graph       │
          │ • Rule 1: Mixer Pattern (Fan-Out/Fan-In)     │
          │ • Rule 2: Burst Transfer (< 20s Velocity)    │
          │ • Rule 3: Sanctioned / Blacklist Counterparty│
          │ • Rule 4: High Priority Fee Front-Running    │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
                 [ DEFENSIVE RISK SYNTHESIS ENGINE ]
          ┌──────────────────────────────────────────────┐
          │ • Calibrated Risk Score (0 - 100)            │
          │ • Threat Tier: CRITICAL / HIGH / MED / LOW   │
          │ • Explainable Risk Breakdown & Attribution   │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
       [ PROFESSIONAL SOC CONSOLE (No Gimmicks / No Gradients) ]
            Streamlit Institutional Security Dashboard
```

---

## 🚀 Quickstart: Running Locally (Windows / Mac / Linux)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Dual Models (Runs in 5 seconds)
```bash
python train.py
```
*Outputs: Trains the XGBoost Classifier and Isolation Forest Anomaly Detector, calculates precision/recall/F1 metrics, and saves model artifacts to `models/`.*

### 3. Launch the Sentinel-Chain Console

#### 🚀 Primary Modern UI (NiceGUI + ECharts + WebSocket Reactive):
```bash
python app.py
```
Open **`http://localhost:8080`** in your browser.
- **Real-Time Reactive Streaming**: Powered by WebSockets (no page reloads, zero flicker)
- **Interactive ECharts UTXO Graph**: Drag nodes, zoom, pan, hover tooltips
- **Radial Risk Score Gauge**: Animated needle with calibrated risk zones
- **Forensic AI Briefing**: Powered by Groq Cloud (`openai/gpt-oss-20b`)
- **Interactive Threat Simulator Sandbox**: Custom slider testbed for instant pre-flight scoring (<15ms)

#### 🛡️ Legacy UI (Streamlit Fallback):
```bash
streamlit run app_streamlit.py
```
Open `http://localhost:8501`.

---

## ☁️ Running on Google Colab (If Judges Ask)

You do **NOT** need to pay for any cloud servers or GPU instances. You can run the entire pipeline in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and select `sentinel_chain_colab.ipynb`.
3. Drag and drop the `.py` files (`app.py`, `train.py`, `detector_engine.py`, `testnet_feed.py`, `data_generator.py`) into the Colab file pane (left sidebar).
4. Run all cells:
   - Cell 1 installs dependencies.
   - Cell 2 trains the models (`!python train.py`).
   - Cell 3 verifies the threat engine.
   - Cell 4 starts Streamlit and generates a public link via `localtunnel` so judges can test it live on their phones or laptops!

---

## 🌐 The Testnet API Question (What to Tell the Judges)

**Judges might ask:** *"Did you connect to Bitcoin Testnet? Which paid API key are you using?"*

**Your Answer:**
> *"We do NOT use paid or proprietary APIs. In true decentralized Bitcoin spirit, we ingest real-time unconfirmed transactions directly from public, open-source Bitcoin Testnet nodes via the **Mempool.space / Blockstream public REST API** (`https://mempool.space/testnet/api/mempool/recent`). It requires zero API keys, no subscription, and complies 100% with the hackathon's requirement of using freely available, open-source tooling."*

---

## 🏆 Hackathon Checklist vs Problem Statement

| Requirement in Problem Statement | Implementation in Sentinel-Chain | File Reference |
| :--- | :--- | :--- |
| **1. Data Layer: Elliptic dataset or Testnet feed** | Built-in Bitcoin Testnet API connector + Elliptic feature benchmark generator | `testnet_feed.py`, `data_generator.py` |
| **2. Dual Model: Anomaly & Classification** | Supervised **XGBoost Classifier** + Unsupervised **Isolation Forest Anomaly Detector** | `train.py`, `detector_engine.py` |
| **3. Real-time alerts on suspicious transfers** | Live alert queue with calculated Risk Scores (0-100) and severity badges | `app.py`, `detector_engine.py` |
| **4. Explainable Risk Reasons** | Concrete reasons outputted for every tx: Mixing patterns, bursts, and risky counterparties | `detector_engine.py` (L125-L165) |
| **5. Graph Analysis & Topology** | NetworkX DAG mapping inputs, transaction nodes, and fan-out distribution | `app.py`, `detector_engine.py` |
| **6. Non-AI Looking SOC Dashboard** | Clean Slate/Steel enterprise theme. Strictly NO purple gradients, NO bouncy AI buttons | `app.py` |

---

## 🎙️ 1-Minute Judge Pitch Script (Team Mavericks)

> *"Good afternoon judges. We are Team Mavericks, presenting **Sentinel-Chain** for Problem #30: Cryptocurrency Transaction Threat Detection.*
> 
> *The central problem in crypto defense is that transfers are irreversible: once a compromised wallet is drained or funds enter a CoinJoin mixer, the victim's money is lost forever within minutes.*
> 
> *Traditional black-box deep learning models fail in compliance because compliance officers cannot file a Suspicious Activity Report (SAR) on an unexplainable confidence score. Our solution combines three layers:*
> 1. *A supervised **XGBoost Classifier** trained on the Elliptic Bitcoin dataset to catch known illicit crime signatures.*
> 2. *An unsupervised **Isolation Forest Anomaly Detector** to flag novel zero-day money laundering topologies.*
> 3. *A deterministic **NetworkX Graph & Rule Engine** that inspects fan-out mixing ratios, burst drain velocity, and hop distance to sanctioned entities.*
> 
> *Every transaction receives an explainable score between 0 and 100 with defensible forensic reasons — for example: 'Mixer pattern with 24 outputs', 'High velocity burst of 4.2 seconds', and 'Direct counterparty flagged as Phishing Drainer'.*
> 
> *Let us demonstrate this live on our dashboard..."*
