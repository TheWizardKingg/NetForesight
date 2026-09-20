from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import requests
from inference_pipeline import analyze_sequence_window, scaler

app = FastAPI(
    title="NETFORESIGHT Engine API",
    description="Dual-Engine Threat Detection & Local LLM Advisory Server"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

def generate_soc_briefing(pipeline_output):
    system_prompt = (
        "You are an expert Security Operations Center (SOC) AI Analyst for NETFORESIGHT. "
        "Your role is to translate raw machine learning threat predictions and SHAP feature "
        "importance scores into concise, actionable advisories for tier-1 security analysts."
    )

    user_prompt = f"""
Analyze the following ML Threat Detection JSON Payload and generate a structured SOC Incident Brief:

[ML METRICS PAYLOAD]
{json.dumps(pipeline_output, indent=2)}

Format your output into these exact sections:
1. Threat Summary: Overview of current detection and overall risk severity.
2. Temporal Forecast (t+1): Explain what the PyTorch Transformer predicts for the next attack step.
3. Feature Attribution (SHAP): Explain the network indicators that triggered this detection.
4. Mitigation Steps: Provide 2 concrete, immediate containment steps for the SOC team.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "No briefing generated.")
        return f"[!] Ollama HTTP Status {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return "[!] Error: Could not connect to Ollama at http://localhost:11434."

class NetworkSequenceInput(BaseModel):
    sequence: list

@app.get("/")
def health_check():
    return {
        "status": "online",
        "system": "NETFORESIGHT AI Engine",
        "models_loaded": ["XGBoost Single-State", "PyTorch Transformer Forecaster", "SHAP Explainer"]
    }

@app.get("/api/sample-predict")
def predict_sample():
    try:
        X_test_seq = np.load("X_sequences.npy")[:1]
        raw_sample = scaler.inverse_transform(X_test_seq[0])

        ml_payload = analyze_sequence_window(raw_sample)
        llm_briefing = generate_soc_briefing(ml_payload)

        return {
            "success": True,
            "metrics": ml_payload,
            "soc_briefing": llm_briefing
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("[+] Starting NETFORESIGHT API Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)