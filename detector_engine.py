"""
Cryptocurrency Threat Detection Engine
Combines:
1. Supervised Classifier (XGBoost) - Known illicit patterns
2. Unsupervised Anomaly Detector (Isolation Forest) - Global structural/topological outliers
3. Entity Behavioral Baseline Profiler - Per-wallet historical anomaly (e.g. 2 BTC -> 299 BTC spike)
4. Graph Topology Analyzer (NetworkX) - Mixing & peeling chain flow
5. Deterministic Cybersecurity Rule Engine - Fast alerts before funds leave
"""

import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx

class ThreatDetectionEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.classifier = None
        self.anomaly_detector = None
        self.scaler = None
        self.feature_cols = None
        self.metrics = None
        
        # In-memory NetworkX transaction graph
        self.tx_graph = nx.DiGraph()
        
        # Blacklist / Sanctioned seed entities (OFAC / Chainalysis sanction lists)
        self.flagged_entities = {
            "bc1q_mixer_wasabi_pool": "Wasabi CoinJoin Mixer Pool",
            "bc1q_lazarus_heist_vault": "Lazarus Group Associated Drainer",
            "1DarknetSilkMarketplace": "Darknet Vendor Deposit Vault",
            "3RansomwareLockBitExtort": "LockBit Ransomware Settlement Address",
            "bc1q_drainer_contract_x": "Phishing Permit2 Drainer Hotwallet"
        }
        
        # Per-Wallet Historical Behavioral Profile Store (Entity Profiling)
        # Tracks past transaction amounts, mean, std, and tx velocity for each wallet
        self.wallet_profiles = {
            "bc1q_retail_user_standard": {
                "tx_history": [1.85, 2.10, 1.95, 2.05, 2.20, 1.90, 2.00],
                "mean_amount": 2.007,
                "std_amount": 0.124,
                "tx_count": 7
            },
            "bc1q_victim_hotwallet_target": {
                "tx_history": [1.5, 2.2, 1.8, 2.1, 2.4],
                "mean_amount": 2.000,
                "std_amount": 0.346,
                "tx_count": 5
            },
            "bc1q_corp_treasury_vault": {
                "tx_history": [5.0, 7.5, 6.0, 5.5],
                "mean_amount": 6.000,
                "std_amount": 1.080,
                "tx_count": 4
            }
        }
        
        self.load_models()

    def load_models(self):
        clf_path = os.path.join(self.models_dir, "classifier.joblib")
        ano_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
        scl_path = os.path.join(self.models_dir, "scaler.joblib")
        feat_path = os.path.join(self.models_dir, "feature_names.joblib")
        met_path = os.path.join(self.models_dir, "metrics.joblib")
        
        if os.path.exists(clf_path) and os.path.exists(ano_path):
            self.classifier = joblib.load(clf_path)
            self.anomaly_detector = joblib.load(ano_path)
            self.scaler = joblib.load(scl_path)
            self.feature_cols = joblib.load(feat_path)
            if os.path.exists(met_path):
                self.metrics = joblib.load(met_path)
            print("[*] ThreatDetectionEngine: Loaded dual models & scaler successfully.")
        else:
            raise FileNotFoundError(f"Models not found in '{self.models_dir}'. Run train.py first.")

    def evaluate_wallet_behavioral_baseline(self, sender_addr, current_amount):
        """
        Entity-Level Behavioral Profiling:
        Checks the sender wallet's individual historical distribution.
        Detects sudden massive deviations (e.g., normally sends 2 BTC, suddenly sends 299 BTC).
        """
        if sender_addr not in self.wallet_profiles:
            # First time seeing this wallet: initialize baseline with this transaction
            self.wallet_profiles[sender_addr] = {
                "tx_history": [current_amount],
                "mean_amount": current_amount,
                "std_amount": max(0.1, current_amount * 0.1),
                "tx_count": 1
            }
            return {
                "is_new_wallet": True,
                "mean_amount": current_amount,
                "std_amount": 0.0,
                "spike_ratio": 1.0,
                "z_score": 0.0,
                "anomaly_triggered": False,
                "reason": None
            }
            
        profile = self.wallet_profiles[sender_addr]
        mean_amt = profile["mean_amount"]
        std_amt = max(0.05, profile["std_amount"])
        
        # Calculate behavioral deviation metrics
        spike_ratio = current_amount / max(0.001, mean_amt)
        z_score = (current_amount - mean_amt) / std_amt
        
        anomaly_triggered = False
        reason = None
        
        # Sudden spike threshold: 5x above historical mean OR Z-score > 4.0
        if spike_ratio >= 5.0 or z_score >= 4.0:
            anomaly_triggered = True
            reason = (
                f"[WALLET BASELINE ANOMALY] Sudden behavioral spike: Transfer of {current_amount:.2f} BTC "
                f"is {spike_ratio:.1f}x higher than this wallet's historical baseline "
                f"(Normal Avg: {mean_amt:.2f} BTC, Std: ±{std_amt:.2f} BTC, Z-Score: +{z_score:.1f}σ). "
                f"High probability of private key theft or account takeover!"
            )
        
        # Update wallet historical profile with rolling window (last 20 txs)
        profile["tx_history"].append(current_amount)
        if len(profile["tx_history"]) > 20:
            profile["tx_history"].pop(0)
            
        profile["mean_amount"] = float(np.mean(profile["tx_history"]))
        profile["std_amount"] = float(np.std(profile["tx_history"]))
        profile["tx_count"] += 1
        
        return {
            "is_new_wallet": False,
            "mean_amount": round(mean_amt, 4),
            "std_amount": round(std_amt, 4),
            "spike_ratio": round(spike_ratio, 2),
            "z_score": round(z_score, 2),
            "tx_count": profile["tx_count"],
            "anomaly_triggered": anomaly_triggered,
            "reason": reason
        }

    def extract_features(self, tx_data):
        """
        Extract numerical features required by XGBoost and Isolation Forest.
        """
        in_degree = int(tx_data.get("in_degree", 1))
        out_degree = int(tx_data.get("out_degree", 2))
        tx_amount = float(tx_data.get("tx_amount", 1.0))
        tx_fee = float(tx_data.get("tx_fee", 0.0001))
        hop_velocity = float(tx_data.get("hop_velocity", 120.0))
        
        fan_out_ratio = out_degree / (in_degree + 1.0)
        fan_in_ratio = in_degree / (out_degree + 1.0)
        is_burst = 1 if (hop_velocity < 25.0) else 0
        
        # Mixing score heuristic
        mixing_score = float(tx_data.get("mixing_score", 0.0))
        if mixing_score == 0.0 and out_degree >= 6 and in_degree <= 2:
            mixing_score = min(1.0, 0.4 + (out_degree * 0.03))
            
        neighbor_avg_amount = float(tx_data.get("neighbor_avg_amount", tx_amount))
        neighbor_illicit_ratio = float(tx_data.get("neighbor_illicit_ratio", 0.0))
        
        feat_dict = {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "tx_amount": tx_amount,
            "tx_fee": tx_fee,
            "hop_velocity": hop_velocity,
            "fan_out_ratio": fan_out_ratio,
            "fan_in_ratio": fan_in_ratio,
            "is_burst": is_burst,
            "mixing_score": mixing_score,
            "neighbor_avg_amount": neighbor_avg_amount,
            "neighbor_illicit_ratio": neighbor_illicit_ratio
        }
        
        # Ensure ordered array matching training columns
        vec = [feat_dict.get(col, 0.0) for col in self.feature_cols]
        return np.array([vec]), feat_dict

    def evaluate_rules(self, tx_data, feat_dict):
        """
        Evaluate deterministic rules mandated by the hackathon problem statement.
        """
        reasons = []
        rule_score = 0.0
        
        sender = tx_data.get("sender", "unknown")
        receiver = tx_data.get("receiver", "unknown")
        outputs = tx_data.get("outputs", [])
        
        # Rule 1: Risky Counterparty Match (OFAC / Darknet / Drainer)
        for entity_addr, entity_name in self.flagged_entities.items():
            if entity_addr in [sender, receiver] or any(entity_addr == out.get("address") for out in outputs):
                rule_score += 40.0
                reasons.append(f"[CRITICAL RULE] Direct counterparty is flagged illicit entity: {entity_name}")
                
        # Rule 2: Mixing Pattern (CoinJoin / Peeling Chain)
        if feat_dict["out_degree"] >= 8 and feat_dict["in_degree"] <= 2:
            rule_score += 25.0
            reasons.append(f"[RULE] Mixer signature detected: High fan-out ({feat_dict['out_degree']} outputs from {feat_dict['in_degree']} inputs)")
        elif feat_dict["mixing_score"] >= 0.70:
            rule_score += 20.0
            reasons.append(f"[RULE] Heuristic mixer score high ({feat_dict['mixing_score']:.2f}): Peeling chain pattern")

        # Rule 3: High Velocity Burst Transfer (Wallet Drainer)
        if feat_dict["hop_velocity"] <= 15.0:
            rule_score += 25.0
            reasons.append(f"[RULE] High-velocity burst: Funds routed in {feat_dict['hop_velocity']:.1f}s (Compromised wallet emptying)")
        elif feat_dict["hop_velocity"] <= 35.0:
            rule_score += 15.0
            reasons.append(f"[RULE] Rapid transfer velocity: {feat_dict['hop_velocity']:.1f}s turnaround")

        # Rule 4: Suspicious Priority Fee Outlier
        if feat_dict["tx_fee"] > 0.01:
            rule_score += 10.0
            reasons.append(f"[RULE] Unusually high priority fee ({feat_dict['tx_fee']:.4f} BTC) to expedite block front-running")

        # Rule 5: Heavy Fan-In Aggregation (Smurfing / Consolidation)
        if feat_dict["in_degree"] >= 8 and feat_dict["out_degree"] <= 2:
            rule_score += 15.0
            reasons.append(f"[RULE] Fund consolidation pattern: {feat_dict['in_degree']} inputs swept into single address")

        return rule_score, reasons

    def update_graph(self, tx_data):
        tx_id = tx_data.get("txId", "tx_unknown")
        sender = tx_data.get("sender", "addr_in")
        receiver = tx_data.get("receiver", "addr_out")
        amount = tx_data.get("tx_amount", 1.0)
        
        self.tx_graph.add_node(sender, type="address")
        self.tx_graph.add_node(tx_id, type="transaction", amount=amount)
        self.tx_graph.add_node(receiver, type="address")
        
        self.tx_graph.add_edge(sender, tx_id, weight=amount)
        self.tx_graph.add_edge(tx_id, receiver, weight=amount)
        
        for out in tx_data.get("outputs", []):
            out_addr = out.get("address", "addr_out")
            out_val = out.get("value", 0.0)
            self.tx_graph.add_node(out_addr, type="address")
            self.tx_graph.add_edge(tx_id, out_addr, weight=out_val)

    def analyze_transaction(self, tx_data):
        """
        Complete 4-Layer Threat Analysis:
        1. Supervised XGBoost Classifier (Known illicit signatures)
        2. Unsupervised Isolation Forest (Global topological outliers)
        3. Wallet Behavioral Profiling (Per-wallet baseline deviation e.g. 2 -> 299 BTC)
        4. NetworkX Graph Topology & Rules (Mixing, bursts, counterparties)
        """
        X_vec, feat_dict = self.extract_features(tx_data)
        X_scaled = self.scaler.transform(X_vec)
        
        sender = tx_data.get("sender", "unknown")
        current_amount = feat_dict["tx_amount"]
        
        # 1. Supervised Model (XGBoost)
        prob_illicit = float(self.classifier.predict_proba(X_scaled)[0, 1])
        
        # 2. Unsupervised Model (Isolation Forest Anomaly)
        raw_anomaly = float(self.anomaly_detector.decision_function(X_scaled)[0])
        anomaly_score = float(1.0 / (1.0 + np.exp(raw_anomaly * 5.0)))
        
        # 3. Entity-Level Wallet Baseline Behavioral Check (The "2 BTC -> 299 BTC" problem!)
        wallet_baseline = self.evaluate_wallet_behavioral_baseline(sender, current_amount)
        
        # 4. Deterministic Rules & Graph Flow
        rule_score, rule_reasons = self.evaluate_rules(tx_data, feat_dict)
        self.update_graph(tx_data)
        
        # If wallet behavioral anomaly triggered, add penalty and reason
        if wallet_baseline["anomaly_triggered"]:
            rule_score += 35.0
            rule_reasons.insert(0, wallet_baseline["reason"])
        
        # 5. Composite Risk Calculation
        composite_risk = (prob_illicit * 30.0) + (anomaly_score * 25.0) + min(45.0, rule_score)
        
        # Escalation conditions
        if wallet_baseline["anomaly_triggered"] and wallet_baseline["spike_ratio"] >= 10.0:
            composite_risk = max(composite_risk, 92.0)
        elif any("CRITICAL RULE" in r for r in rule_reasons):
            composite_risk = max(composite_risk, 88.0)
        elif prob_illicit > 0.85 and feat_dict["is_burst"] == 1:
            composite_risk = max(composite_risk, 78.0)
            
        final_risk_score = round(float(np.clip(composite_risk, 0.0, 100.0)), 1)
        
        # Threat Level
        if final_risk_score >= 75.0:
            risk_level = "CRITICAL"
            badge_color = "#E53E3E"
        elif final_risk_score >= 50.0:
            risk_level = "HIGH"
            badge_color = "#DD6B20"
        elif final_risk_score >= 25.0:
            risk_level = "MEDIUM"
            badge_color = "#D69E2E"
        else:
            risk_level = "LOW"
            badge_color = "#38A169"
            
        all_reasons = list(rule_reasons)
        if prob_illicit >= 0.70:
            all_reasons.append(f"[ML CLASSIFIER] XGBoost predicts {prob_illicit*100:.1f}% illicit probability based on historical Elliptic signatures")
        elif prob_illicit <= 0.15:
            all_reasons.append(f"[ML CLASSIFIER] Normal transaction profile ({(1-prob_illicit)*100:.1f}% licit confidence)")
            
        if anomaly_score >= 0.70:
            all_reasons.append(f"[GLOBAL ANOMALY] Isolation Forest detected severe graph topological outlier ({anomaly_score*100:.1f}% deviation from standard distribution)")
        elif anomaly_score >= 0.50:
            all_reasons.append(f"[GLOBAL ANOMALY] Moderate structural outlier in degree or fee-to-amount ratio")

        if not all_reasons:
            all_reasons.append("Standard retail/exchange transaction parameters within baseline tolerances")
            
        return {
            "txId": tx_data.get("txId", "unknown"),
            "risk_score": final_risk_score,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "prob_illicit": round(prob_illicit * 100, 1),
            "anomaly_score": round(anomaly_score * 100, 1),
            "rule_score": round(rule_score, 1),
            "features": feat_dict,
            "reasons": all_reasons,
            "wallet_baseline": wallet_baseline,
            "timestamp": tx_data.get("timestamp", "Just now"),
            "amount_btc": feat_dict["tx_amount"],
            "fee_btc": feat_dict["tx_fee"],
            "sender": sender,
            "receiver": tx_data.get("receiver", "bc1q_unnamed")
        }
