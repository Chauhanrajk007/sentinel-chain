"""
Update the master notebook with the EXACT plain-English forensic labels for the top 10 features:
- feat_90: Fee Density Ratio (Priority Overpay)
- feat_53: Peeling Chain Remainder (Small Change)
- feat_132: 1-Hop Neighbor Min Amount (Micro-Dusting)
- feat_59: Input Dominance Ratio (Single-Source Sweep)
- feat_163: Flow-Through Velocity (Transit Mule Pipe)
- feat_46: Input Amount Variance (Erratic Funding)
- feat_49: Fee Spread / Inconsistency (Dynamic Bidding)
- feat_5: Output Count (Fan-Out / Mixer Indicator)
- feat_152: Neighbor Max Fee (Panic / Front-Run Fee)
- feat_65: Output Uniformity (CoinJoin Denominations)
"""

import json

with open("sentinel_chain_colab_optimized.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update the feature mapping dictionary in cell 15 (STEP 7)
forensic_mapping_code = """# ==============================================================================
# EXACT FORENSIC TRANSLATION MAP FOR ELLIPTIC BITCOIN FEATURES
# Translates abstract codes into concrete financial crime indicators
# ==============================================================================

FEATURE_NAME_MAP = {
    'feat_90': 'Fee Density Ratio (Priority Overpay)',
    'feat_53': 'Peeling Chain Remainder (Small Change)',
    'feat_132': '1-Hop Neighbor Min Amount (Micro-Dusting)',
    'feat_59': 'Input Dominance Ratio (Single-Source Sweep)',
    'feat_163': 'Flow-Through Velocity (Transit Mule Pipe)',
    'feat_46': 'Input Amount Variance (Erratic Funding)',
    'feat_49': 'Fee Spread / Inconsistency (Dynamic Bidding)',
    'feat_5': 'Output Count (Mixer Fan-Out Indicator)',
    'feat_152': 'Neighbor Max Fee (Panic / Front-Run Fee)',
    'feat_65': 'Output Uniformity (CoinJoin Denominations)',
    'feat_1': 'Time-Step (Block Window 1-49)',
    'feat_2': 'Miner Transaction Fee (BTC)',
    'feat_3': 'Total Transferred Amount (BTC)',
    'feat_4': 'Input Count (In-Degree)',
    'feat_6': 'Fan-Out Ratio (out / in)',
    'feat_7': 'Fan-In Ratio (in / out)',
    'feat_8': 'Fee-to-Amount Ratio',
    'feat_9': 'Hop Velocity (Turnaround Seconds)',
    'feat_10': 'Transaction Size (vBytes)',
    'feat_11': 'Output Variance & Skewness',
    'feat_94': '1-Hop Neighbor Mean Amount (Graph Context)',
    'feat_95': '1-Hop Neighbor Mean Fee (Graph Context)',
    'feat_111': '1-Hop Neighbor StdDev Amount (Graph Context)',
    'feat_126': '1-Hop Neighbor Min Amount (Graph Context)',
    'feat_141': '1-Hop Neighbor Max Amount (Graph Context)',
    'feat_156': 'Graph Backward Flow Ratio (Graph Context)'
}

importances = clf.feature_importances_
top_k = 10
sorted_idx = np.argsort(importances)[::-1][:top_k]

sample_illicit_tx = X_test[y_test == 1][0]
table_rows = []
plot_labels = []
plot_weights = []

for idx in sorted_idx:
    f_code = feature_cols[idx]
    f_name = FEATURE_NAME_MAP.get(f_code, f'1-Hop Graph Topology Feature ({f_code})')
    weight = importances[idx] * 100
    raw_val = sample_illicit_tx[idx]
    
    plot_labels.append(f'{f_name} [{f_code}]')
    plot_weights.append(importances[idx])
    
    table_rows.append({
        'Feature Code': f_code,
        'Forensic Crime Meaning': f_name,
        'Actual Value in Sample': f'{raw_val:.4f}',
        'Detection Impact': f'{weight:.2f}%'
    })

print('='*90)
print('EXACT TOP FEATURES & FORENSIC CRIME MEANINGS (FROM OFFICIAL DATASET)')
print('='*90)
display(pd.DataFrame(table_rows))

# Horizontal Bar Chart with Plain-English Forensic Labels
plt.figure(figsize=(11, 5))
plt.barh(plot_labels[::-1], plot_weights[::-1], color='#2563eb')
plt.title('Top 10 Most Discriminative Blockchain Threat Indicators', fontsize=12, fontweight='bold')
plt.xlabel('XGBoost Relative Feature Importance Weight')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""

# Replace in notebook (Cell 15 is Step 7)
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "STEP 7: Real-World Feature Mapping" in source:
        # The next code cell is the target
        nb["cells"][i+1]["source"] = [forensic_mapping_code]
        print(f"Updated cell {i+1} with exact forensic labels!")
        break

with open("sentinel_chain_colab_optimized.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("[OK] sentinel_chain_colab_optimized.ipynb updated successfully!")
