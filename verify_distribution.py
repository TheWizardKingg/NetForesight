import numpy as np
import joblib
from inference_pipeline import analyze_sequence_window

print("\n[+] Filtering for a high-confidence attack window...")

X_seq = np.load("X_sequences.npy")
y_targets = np.load("y_targets.npy")
xgb_model = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("feature_scaler.pkl")

# Flatten sequence windows for XGBoost evaluation
X_flat = X_seq.reshape(len(X_seq), -1)

# Get full class probabilities across all 1M rows
probs = xgb_model.predict_proba(X_flat)

# Find indices where non-benign (class 1 or 2) probability is > 85%
non_benign_probs = probs[:, 1:]  # FTP-BruteForce and SSH-Bruteforce probabilities
high_conf_indices = np.where(np.max(non_benign_probs, axis=1) > 0.85)[0]

print(f"[+] Total high-confidence attack windows (>85% probability): {len(high_conf_indices)}")

if len(high_conf_indices) > 0:
    # Pick the top high-confidence attack sample
    target_idx = high_conf_indices[0]
    raw_seq = scaler.inverse_transform(X_seq[target_idx])
    
    # Pass through full dual-engine pipeline
    result = analyze_sequence_window(raw_seq)

    print("\n" + "="*60)
    print("        CONFIRMED HIGH-CONFIDENCE ATTACK PIPELINE        ")
    print("="*60)
    print(f"Dataset Ground Truth       : {label_encoder.inverse_transform([y_targets[target_idx]])[0]}")
    print(f"Current State (XGBoost)     : {result['current_detection']['predicted_attack']} ({result['current_detection']['confidence']}% confidence)")
    print(f"Next-Stage Forecast (Trans) : {result['next_stage_forecast']['predicted_next_attack']} ({result['next_stage_forecast']['probability']}% probability)")
    print(f"Calculated Risk Score       : {result['risk_assessment']['score']} / 100 ({result['risk_assessment']['severity']})")
    print("\nTop SHAP Triggers:")
    for t in result['top_shap_triggers']:
        print(f"  - {t['feature']}: {t['value']} (Weight: {t['importance_score']:.4f})")
    print("="*60 + "\n")