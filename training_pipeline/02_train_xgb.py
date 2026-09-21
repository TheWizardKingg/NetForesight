import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("--- BOOTING XGBOOST TRAINING ---")
# We only need the last frame of each sequence to predict the current stage
X_seq = np.load('training_pipeline/X_seq.npy')
y_curr = np.load('training_pipeline/y_curr.npy')
encoder = joblib.load('training_pipeline/label_encoder.pkl')

# Flatten the last frame of the window for XGBoost (batch_size, features)
X_flat = X_seq[:, -1, :] 

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_curr, test_size=0.2, random_state=42)

# Hardcore parameters for raw accuracy
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method='hist', # Fast and heavy
    n_jobs=-1
)

print("Training XGBoost... don't touch anything.")
xgb.fit(X_train, y_train)

print("\n--- XGBoost Performance ---")
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

joblib.dump(xgb, 'xgboost_model.pkl')
print("Saved xgboost_model.pkl")