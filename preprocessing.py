import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ---------------------------------------------------------
# 1. Configuration & Core Feature Mapping
# ---------------------------------------------------------
INPUT_CSV = "data/cic.csv"  # Check your exact CSV path in data/
CLEAN_CSV = "cleaned_cic_selected.csv"
SEQUENCE_LENGTH = 5    # Number of prior flow events per window (t-4 ... t)

SELECTED_COLUMNS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", 
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "SYN Flag Cnt", "RST Flag Cnt", 
    "ACK Flag Cnt", "Label"
]

MITRE_MAP = {
    "Benign": "Reconnaissance / Normal",
    "Bot": "Command and Control",
    "FTP-BruteForce": "Credential Access",
    "SSH-Bruteforce": "Credential Access",
    "DoS attacks-GoldenEye": "Impact",
    "DoS attacks-Slowloris": "Impact",
    "DoS attacks-SlowHTTPTest": "Impact",
    "DoS attacks-Hulk": "Impact",
    "DDOS attack-HOIC": "Impact",
    "DDOS attack-LOIC-UDP": "Impact",
    "DDOS attack-LOIC-HTTP": "Impact",
    "Brute Force -Web": "Initial Access",
    "Brute Force -XSS": "Initial Access",
    "SQL Injection": "Initial Access",
    "Infiltration": "Lateral Movement"
}

print("[+] Phase 1 Started: Processing raw dataset...")

# ---------------------------------------------------------
# 2. Polars Ingestion & Cleaning
# ---------------------------------------------------------
df = pl.read_csv(INPUT_CSV, columns=SELECTED_COLUMNS)

# Parse Timestamps and clean Infinite / NaN rates
df = df.with_columns(
    pl.col("Timestamp").str.to_datetime("%d/%m/%Y %H:%M:%S", strict=False)
)

df = df.with_columns([
    pl.when(pl.col("Flow Byts/s").is_infinite() | pl.col("Flow Byts/s").is_nan())
      .then(0)
      .otherwise(pl.col("Flow Byts/s"))
      .alias("Flow Byts/s"),
    pl.when(pl.col("Flow Pkts/s").is_infinite() | pl.col("Flow Pkts/s").is_nan())
      .then(0)
      .otherwise(pl.col("Flow Pkts/s"))
      .alias("Flow Pkts/s")
]).drop_nulls()

# Sort chronologically to preserve attack timelines
df = df.sort("Timestamp")

# Use replace_strict for updated Polars API
df = df.with_columns(
    pl.col("Label").replace_strict(MITRE_MAP, default="Unknown").alias("Mitre_Stage")
)

print(f"[+] Cleaned dataset shape: {df.shape}")

# ---------------------------------------------------------
# 3. Label Encoding & Feature Normalization
# ---------------------------------------------------------
pdf = df.to_pandas()

# Encode classification labels
label_encoder = LabelEncoder()
pdf["Target_Code"] = label_encoder.fit_transform(pdf["Label"])
joblib.dump(label_encoder, "label_encoder.pkl")

# Identify numerical features for model input
feature_cols = [col for col in SELECTED_COLUMNS if col not in ["Timestamp", "Label"]]

# Scale numerical columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pdf[feature_cols])
joblib.dump(scaler, "feature_scaler.pkl")

print("[+] Features extracted and normalized successfully.")

# ---------------------------------------------------------
# 4. Sequence Builder (Temporal Windows)
# ---------------------------------------------------------
print(f"[+] Constructing sequence windows (Window Size = {SEQUENCE_LENGTH})...")

target_array = pdf["Target_Code"].values
n_samples = len(scaled_features) - SEQUENCE_LENGTH

X_seq = np.empty((n_samples, SEQUENCE_LENGTH, scaled_features.shape[1]), dtype=np.float32)
y_seq = target_array[SEQUENCE_LENGTH:]

for i in range(n_samples):
    X_seq[i] = scaled_features[i : i + SEQUENCE_LENGTH]

# Save processed binaries for downstream modeling
np.save("X_sequences.npy", X_seq)
np.save("y_targets.npy", y_seq)

print(f"[+] Phase 1 Complete! Output shapes:")
print(f"    - Sequences (X): {X_seq.shape}")
print(f"    - Targets   (y): {y_seq.shape}")