"""
SENTINEL-CHAIN: RESTful Threat Detection API
Built with FastAPI for automated exchange/wallet integration.
Allows external wallets, exchanges, and smart contracts to query threat risk in real-time (<15ms).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from detector_engine import ThreatDetectionEngine

app = FastAPI(
    title="SENTINEL-CHAIN Threat Detection API",
    description="Automated Real-Time Crypto Threat Scoring & Graph Reasoning API (AVISHKAR'26 | Team Mavericks)",
    version="1.0.0"
)

# Initialize engine on startup
engine = ThreatDetectionEngine()

class TransactionRequest(BaseModel):
    txId: Optional[str] = Field("tx_api_query", description="Transaction Hash")
    sender: str = Field("bc1q_sender", description="Sender wallet address")
    receiver: str = Field("bc1q_receiver", description="Receiver wallet address")
    amount_btc: float = Field(..., gt=0, description="Transfer amount in BTC")
    fee_btc: Optional[float] = Field(0.0001, description="Transaction fee in BTC")
    in_degree: Optional[int] = Field(1, description="Number of inputs")
    out_degree: Optional[int] = Field(2, description="Number of outputs")
    hop_velocity: Optional[float] = Field(120.0, description="Time since last transaction in seconds")
    mixing_score: Optional[float] = Field(0.0, description="Optional heuristic mixing indicator")

class ThreatAnalysisResponse(BaseModel):
    txId: str
    risk_score: float
    risk_level: str
    badge_color: str
    prob_illicit: float
    anomaly_score: float
    rule_score: float
    reasons: List[str]
    wallet_baseline: dict
    timestamp: str

@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "SENTINEL-CHAIN Crypto Threat Engine",
        "team": "Mavericks (AV26-914D8EC853)",
        "docs_url": "/docs"
    }

@app.post("/api/v1/detect", response_model=ThreatAnalysisResponse)
def detect_threat(req: TransactionRequest):
    """
    Real-Time Threat Detection Endpoint.
    Analyzes transaction through XGBoost Classifier, Isolation Forest, and Rule Engine.
    """
    try:
        tx_data = {
            "txId": req.txId,
            "sender": req.sender,
            "receiver": req.receiver,
            "tx_amount": req.amount_btc,
            "tx_fee": req.fee_btc,
            "in_degree": req.in_degree,
            "out_degree": req.out_degree,
            "hop_velocity": req.hop_velocity,
            "mixing_score": req.mixing_score
        }
        result = engine.analyze_transaction(tx_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
