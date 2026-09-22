import torch
import xgboost as xgb
import joblib
import numpy as np

print("[+] Training XGBoost on single-step UNSW features...")
data = torch.load("data/unsw_sequences.pt")
X_seq = data["X"].numpy()  # (N, 5, num_features)
y = data["y"].numpy()

# Train XGBoost on the latest frame in the sequence (index -1)
X_single = X_seq[:, -1, :]

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    random_state=42
)
model.fit(X_single, y)

joblib.dump(model, "backend/xgboost_model.pkl")
print("[✔] Saved backend/xgboost_model.pkl")