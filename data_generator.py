"""
Cryptocurrency Transaction Threat Detection - Synthetic & Elliptic Data Generator
Simulates realistic Bitcoin transaction graphs conforming to the Elliptic Data Benchmark.
Used for offline hackathon training and automated benchmarking.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_elliptic_dataset(n_samples=5000, random_state=42, output_dir="data"):
    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)
    
    # 70% Licit, 15% Illicit, 15% Unknown/Zero-Day
    n_licit = int(n_samples * 0.70)
    n_illicit = int(n_samples * 0.15)
    n_unknown = n_samples - n_licit - n_illicit
    
    records = []
    
    # 1. Generate Licit Transactions (Exchanges, Miners, Standard P2P)
    for i in range(n_licit):
        tx_id = f"tx_licit_{i:06d}"
        time_step = np.random.randint(1, 50)
        in_degree = int(np.clip(np.random.poisson(lam=1.8), 1, 6))
        out_degree = int(np.clip(np.random.poisson(lam=2.1), 1, 4)) # typically 2 (payment + change)
        tx_amount = float(np.clip(np.random.lognormal(mean=0.2, sigma=1.1), 0.001, 80.0))
        tx_fee = float(np.clip(tx_amount * np.random.uniform(0.0001, 0.001), 0.00001, 0.05))
        hop_velocity = float(np.random.exponential(scale=600)) # moderate delay between txs (minutes)
        fan_out_ratio = out_degree / (in_degree + 1.0)
        fan_in_ratio = in_degree / (out_degree + 1.0)
        is_burst = 1 if (hop_velocity < 20 and np.random.rand() < 0.05) else 0
        mixing_score = float(np.random.beta(a=1, b=8)) # low mixer pattern
        neighbor_avg_amount = float(tx_amount * np.random.uniform(0.7, 1.3))
        neighbor_illicit_ratio = float(np.random.beta(a=1, b=20)) # clean counterparties
        
        records.append({
            "txId": tx_id,
            "time_step": time_step,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "tx_amount": round(tx_amount, 6),
            "tx_fee": round(tx_fee, 6),
            "hop_velocity": round(hop_velocity, 2),
            "fan_out_ratio": round(fan_out_ratio, 4),
            "fan_in_ratio": round(fan_in_ratio, 4),
            "is_burst": is_burst,
            "mixing_score": round(mixing_score, 4),
            "neighbor_avg_amount": round(neighbor_avg_amount, 6),
            "neighbor_illicit_ratio": round(neighbor_illicit_ratio, 4),
            "class": "2" # Licit
        })
        
    # 2. Generate Illicit Transactions (Mixers, Ransomware, Peeling Chains, Stolen Wallet Drains)
    for i in range(n_illicit):
        tx_id = f"tx_illicit_{i:06d}"
        time_step = np.random.randint(1, 50)
        threat_type = np.random.choice(["mixer", "burst_drain", "peeling_chain"])
        
        if threat_type == "mixer":
            # 1 input split into 10-30 identical micro-outputs (Wasabi / CoinJoin)
            in_degree = 1
            out_degree = int(np.random.randint(8, 35))
            tx_amount = float(np.random.uniform(2.0, 150.0))
            hop_velocity = float(np.random.uniform(5.0, 45.0)) # automated rapid split
            mixing_score = float(np.random.uniform(0.75, 0.99))
            is_burst = 1
        elif threat_type == "burst_drain":
            # compromised wallet emptied within 60 seconds
            in_degree = int(np.random.randint(3, 10))
            out_degree = 1
            tx_amount = float(np.random.uniform(5.0, 250.0))
            hop_velocity = float(np.random.uniform(1.0, 15.0)) # rapid drain
            mixing_score = float(np.random.uniform(0.3, 0.7))
            is_burst = 1
        else: # peeling chain
            in_degree = 1
            out_degree = 2
            tx_amount = float(np.random.uniform(10.0, 80.0))
            hop_velocity = float(np.random.uniform(10.0, 60.0))
            mixing_score = float(np.random.uniform(0.6, 0.85))
            is_burst = 0
            
        tx_fee = float(np.clip(tx_amount * np.random.uniform(0.002, 0.01), 0.0005, 0.2)) # high priority fee
        fan_out_ratio = out_degree / (in_degree + 1.0)
        fan_in_ratio = in_degree / (out_degree + 1.0)
        neighbor_avg_amount = float(tx_amount * np.random.uniform(0.8, 1.5))
        neighbor_illicit_ratio = float(np.random.beta(a=6, b=2)) # dirty counterparties
        
        records.append({
            "txId": tx_id,
            "time_step": time_step,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "tx_amount": round(tx_amount, 6),
            "tx_fee": round(tx_fee, 6),
            "hop_velocity": round(hop_velocity, 2),
            "fan_out_ratio": round(fan_out_ratio, 4),
            "fan_in_ratio": round(fan_in_ratio, 4),
            "is_burst": is_burst,
            "mixing_score": round(mixing_score, 4),
            "neighbor_avg_amount": round(neighbor_avg_amount, 6),
            "neighbor_illicit_ratio": round(neighbor_illicit_ratio, 4),
            "class": "1" # Illicit
        })
        
    # 3. Generate Unknown Transactions (Novel zero-day patterns for anomaly evaluation)
    for i in range(n_unknown):
        tx_id = f"tx_unknown_{i:06d}"
        time_step = np.random.randint(1, 50)
        in_degree = int(np.clip(np.random.poisson(lam=2.5), 1, 15))
        out_degree = int(np.clip(np.random.poisson(lam=3.0), 1, 20))
        tx_amount = float(np.clip(np.random.lognormal(mean=0.5, sigma=1.5), 0.01, 150.0))
        tx_fee = float(tx_amount * np.random.uniform(0.0001, 0.005))
        hop_velocity = float(np.random.exponential(scale=300))
        fan_out_ratio = out_degree / (in_degree + 1.0)
        fan_in_ratio = in_degree / (out_degree + 1.0)
        is_burst = 1 if (hop_velocity < 30) else 0
        mixing_score = float(np.random.uniform(0.1, 0.8))
        neighbor_avg_amount = float(tx_amount * np.random.uniform(0.5, 1.8))
        neighbor_illicit_ratio = float(np.random.uniform(0.0, 0.5))
        
        records.append({
            "txId": tx_id,
            "time_step": time_step,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "tx_amount": round(tx_amount, 6),
            "tx_fee": round(tx_fee, 6),
            "hop_velocity": round(hop_velocity, 2),
            "fan_out_ratio": round(fan_out_ratio, 4),
            "fan_in_ratio": round(fan_in_ratio, 4),
            "is_burst": is_burst,
            "mixing_score": round(mixing_score, 4),
            "neighbor_avg_amount": round(neighbor_avg_amount, 6),
            "neighbor_illicit_ratio": round(neighbor_illicit_ratio, 4),
            "class": "unknown" # Unlabeled
        })
        
    df = pd.DataFrame(records).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Save features and classes separately to mimic official Elliptic structure
    classes_df = df[["txId", "class"]]
    features_df = df.drop(columns=["class"])
    
    features_path = os.path.join(output_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(output_dir, "elliptic_txs_classes.csv")
    
    features_df.to_csv(features_path, index=False)
    classes_df.to_csv(classes_path, index=False)
    print(f"[OK] Generated {len(df)} transactions into '{features_path}' & '{classes_path}'")
    return features_path, classes_path

if __name__ == "__main__":
    generate_synthetic_elliptic_dataset()
