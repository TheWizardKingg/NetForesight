import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt

print("[+] Phase 2 Started: Training XGBoost Baseline Model...")

# 1. Load sequence binaries generated from Phase 1
X_seq = np.load("X_sequences.npy")
y_seq = np.load("y_targets.npy")

# Flatten the sequence window for tabular gradient boosting (samples, features * sequence_length)
n_samples, seq_len, n_features = X_seq.shape
X_flat = X_seq.reshape(n_samples, seq_len * n_features)

# Load label encoder for stage names
label_encoder = joblib.load("label_encoder.pkl")
class_names = label_encoder.classes_

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"[+] Training data size: {X_train.shape[0]} | Testing data size: {X_test.shape[0]}")

# 3. Instantiate and Train XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",  # High-speed CPU histogram optimization
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save the trained model artifact
joblib.dump(model, "xgboost_model.pkl")
print("[+] XGBoost Model trained and saved as 'xgboost_model.pkl'!")

# 4. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[===] Model Accuracy: {acc * 100:.2f}% [===]\n")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# 5. SHAP Explainability Feature Calculation
print("[+] Calculating SHAP values for SOC Explainability Dashboard...")
# Sample 500 rows for high-speed SHAP computation
sample_idx = np.random.choice(X_test.shape[0], 500, replace=False)
X_shap_sample = X_test[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_shap_sample)

# Save explainer object
joblib.dump(explainer, "shap_explainer.pkl")
print("[+] SHAP Explainer saved as 'shap_explainer.pkl'!")