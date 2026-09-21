import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("--- BOOTING PREPROCESSING ---")

# Forward slashes prevent Windows backslash escape errors
data_path = 'data/cic.csv'
if not os.path.exists(data_path):
    data_path = '../data/cic.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Can't find {data_path}. Check your path, bro.")

print(f"Loading dataset from {data_path}...")
df = pd.read_csv(data_path)

# 1. Clean inf/NaN values from raw CIC-IDS network captures
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 2. Drop non-numeric identifier columns
cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Extract features & clip extreme numerical values to fit float32
X_raw = df.drop(columns=['Label']).values.astype(np.float64)
y_raw = df['Label'].values
X_raw = np.clip(X_raw, -1e9, 1e9).astype(np.float32)

# 4. Fit Scaler and Encoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)

joblib.dump(scaler, 'training_pipeline/feature_scaler.pkl')
joblib.dump(encoder, 'training_pipeline/label_encoder.pkl')

print(f"Mapped Classes: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

# 5. Generate Sequence Windows (Filtering Fake Drop-Offs)
window_size = 5
X_seq, y_curr, y_next = [], [], []

print("Generating sequence windows and stripping artificial boundaries...")
for i in range(len(X_scaled) - window_size):
    current_window_labels = y_raw[i : i + window_size]
    next_label = y_raw[i + window_size]
    
    # Drop windows where 100% attack abruptly cuts to Benign due to capture limits
    if np.all(current_window_labels != 'Benign') and next_label == 'Benign':
        continue
        
    X_seq.append(X_scaled[i : i + window_size])
    y_curr.append(y_encoded[i + window_size - 1])
    y_next.append(y_encoded[i + window_size])

X_seq = np.array(X_seq, dtype=np.float32)
y_curr = np.array(y_curr, dtype=np.int64)
y_next = np.array(y_next, dtype=np.int64)

np.save('training_pipeline/X_seq.npy', X_seq)
np.save('training_pipeline/y_curr.npy', y_curr)
np.save('training_pipeline/y_next.npy', y_next)

print(f"Done. Successfully saved {len(X_seq)} sequence samples.")