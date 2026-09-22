import numpy as np
import joblib
import shap
from xgboost import XGBClassifier

print("[*] Training XGBoost on CPU (Max Threads)...")
X_train = np.load("training_pipeline/tmp/X_train.npy")
y_train = np.load("training_pipeline/tmp/y_train.npy")

xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    n_jobs=-1,  
    tree_method="hist"
)
xgb_model.fit(X_train, y_train)

print("[*] Generating SHAP Explainer...")
background_sample = shap.sample(X_train, 100)
shap_explainer = shap.TreeExplainer(xgb_model, background_sample, feature_perturbation="interventional")

joblib.dump(xgb_model, "backend/xgboost_model.pkl")
joblib.dump(shap_explainer, "backend/shap_explainer.pkl")
print("[+] XGBoost & SHAP Complete. Deployed to backend/.")