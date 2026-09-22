import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Strict 16 feature enforcement
FEATURE_NAMES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt"
]

print("[*] Running Data Prep...")
# CHANGE THIS IF YOUR CSV IS NAMED DIFFERENTLY
df = pd.read_csv("data/cic.csv") 

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X_raw = df[FEATURE_NAMES]
y_raw = df["Label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

def create_sequences(data, labels, seq_len=5):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:(i + seq_len)])
        ys.append(labels[i + seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_encoded)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Auto-create directories
os.makedirs("backend", exist_ok=True)
os.makedirs("training_pipeline/tmp", exist_ok=True)

# Drop artifacts directly where your API needs them
joblib.dump(scaler, "backend/feature_scaler.pkl")
joblib.dump(label_encoder, "backend/label_encoder.pkl")

np.save("training_pipeline/tmp/X_train.npy", X_train)
np.save("training_pipeline/tmp/y_train.npy", y_train)
np.save("training_pipeline/tmp/X_seq_train.npy", X_seq_train)
np.save("training_pipeline/tmp/y_seq_train.npy", y_seq_train)

print("[+] Data Prep Complete. Scaler & Encoder deployed to backend/.")