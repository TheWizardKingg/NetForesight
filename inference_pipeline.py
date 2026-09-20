import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("[+] Initializing NETFORESIGHT Unified Prediction Pipeline...")

# ---------------------------------------------------------
# 1. Load Artifacts & Encoders
# ---------------------------------------------------------
scaler = joblib.load("feature_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
xgboost_model = joblib.load("xgboost_model.pkl")
shap_explainer = joblib.load("shap_explainer.pkl")

class_names = label_encoder.classes_
num_classes = len(class_names)

FEATURE_NAMES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt"
]

# ---------------------------------------------------------
# 2. Re-instantiate PyTorch Transformer Architecture
# ---------------------------------------------------------
class AttackForecasterTransformer(nn.Module):
    def __init__(self, feature_dim=16, seq_len=5, num_classes=3, d_model=64, nhead=4, num_layers=2):
        super(AttackForecasterTransformer, self).__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_out(x)

transformer_model = AttackForecasterTransformer(feature_dim=len(FEATURE_NAMES), seq_len=5, num_classes=num_classes)
transformer_model.load_state_dict(torch.load("transformer_forecaster.pt", map_location=torch.device('cpu')))
transformer_model.eval()

print("[+] All ML Models, Scalers, and Explainers loaded successfully!")

# ---------------------------------------------------------
# 3. Core Inference & Risk Scoring Function
# ---------------------------------------------------------
def analyze_sequence_window(sequence_window_raw):
    """
    Input: numpy array of shape (5, 16) containing unscaled numeric feature values for 5 consecutive flows
    """
    # Wrap in DataFrame to maintain feature names and suppress Sklearn warning
    df_raw = pd.DataFrame(sequence_window_raw, columns=FEATURE_NAMES)
    sequence_scaled = scaler.transform(df_raw)  # Shape: (5, 16)
    
    # --- A. XGBoost Current-State Classification ---
    xgb_input = sequence_scaled.reshape(1, -1)
    
    current_class_idx = xgboost_model.predict(xgb_input)[0]
    current_class_label = class_names[current_class_idx]
    current_probs = xgboost_model.predict_proba(xgb_input)[0]
    current_confidence = float(np.max(current_probs))
    
    # --- B. PyTorch Transformer Next-Stage Forecast (t+1) ---
    tensor_input = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 5, 16)
    with torch.no_grad():
        logits = transformer_model(tensor_input)
        forecast_probs = torch.softmax(logits, dim=1).numpy()[0]
        
    forecast_class_idx = np.argmax(forecast_probs)
    forecast_class_label = class_names[forecast_class_idx]
    forecast_confidence = float(forecast_probs[forecast_class_idx])
    
    # --- C. SHAP Feature Attribution ---
    shap_vals = shap_explainer(xgb_input)
    
    if len(shap_vals.values.shape) == 3:  # Multi-class SHAP output
        feature_importance = np.abs(shap_vals.values[0, :, current_class_idx])
    else:
        feature_importance = np.abs(shap_vals.values[0])
        
    feature_importance_reshaped = feature_importance.reshape(5, 16).mean(axis=0)
    top_feature_indices = np.argsort(feature_importance_reshaped)[::-1][:3]
    
    top_triggers = [
        {
            "feature": FEATURE_NAMES[idx],
            "value": round(float(sequence_window_raw[-1][idx]), 2),
            "importance_score": float(feature_importance_reshaped[idx])
        }
        for idx in top_feature_indices
    ]
    
    # --- D. Dynamic Risk Score Calculation (0 - 100) ---
    base_risk = 10 if current_class_label == "Benign" else 70
    threat_multiplier = 1.0 if current_class_label == "Benign" else 1.3
    risk_score = min(100, int((base_risk * current_confidence * threat_multiplier) + (forecast_confidence * 15)))
    
    payload = {
        "current_detection": {
            "predicted_attack": current_class_label,
            "confidence": round(current_confidence * 100, 2)
        },
        "next_stage_forecast": {
            "predicted_next_attack": forecast_class_label,
            "probability": round(forecast_confidence * 100, 2)
        },
        "risk_assessment": {
            "score": risk_score,
            "severity": "CRITICAL" if risk_score > 75 else ("MEDIUM" if risk_score > 45 else "LOW")
        },
        "top_shap_triggers": top_triggers
    }
    
    return payload

if __name__ == "__main__":
    X_test_seq = np.load("X_sequences.npy")[:1]
    raw_sample = scaler.inverse_transform(X_test_seq[0])
    
    result = analyze_sequence_window(raw_sample)
    print("\n[===] Unified Pipeline JSON Output [===]")
    print(json.dumps(result, indent=2))