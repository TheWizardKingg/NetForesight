import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def process_unsw_dataset():
    print("[+] Loading UNSW-NB15 train and test sets...")
    train_path = "data/UNSW_NB15_training-set.csv"
    test_path = "data/UNSW_NB15_testing-set.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("[-] Missing UNSW CSV files in data/ directory!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Merge for chronological sequence building
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Target Column: attack_cat
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
        df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})
        target_col = "attack_cat"
    else:
        target_col = "label"

    # Encode Multi-class Target Labels
    le = LabelEncoder()
    df["target"] = le.fit_transform(df[target_col])
    
    os.makedirs("backend", exist_ok=True)
    joblib.dump(le, "backend/label_encoder.pkl")
    print(f"[+] Encoded {len(le.classes_)} Classes: {list(le.classes_)}")

    # Encode categorical features
    cat_cols = ["proto", "service", "state"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Select numerical feature columns (Drop ID and target columns)
    ignore_cols = ["id", "label", "attack_cat", "target"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    print(f"[+] Extracted {len(feature_cols)} network features.")

    # Clean NaN / Inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    joblib.dump(scaler, "backend/feature_scaler.pkl")
    joblib.dump(feature_cols, "backend/feature_names.pkl")

    y = df["target"].values

    # Build 5-step Sliding Sequences
    sequence_length = 5
    print(f"[+] Building 5-step sliding window sequence tensors...")
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)

    # Downsample majority Normal class to prevent model bias
    normal_code = le.transform(["Normal"])[0] if "Normal" in le.classes_ else 0
    normal_idx = np.where(y_seq == normal_code)[0]
    attack_idx = np.where(y_seq != normal_code)[0]

    max_normal = int(len(attack_idx) * 1.2)
    if len(normal_idx) > max_normal:
        np.random.seed(42)
        normal_idx = np.random.choice(normal_idx, size=max_normal, replace=False)

    balanced_idx = np.concatenate([attack_idx, normal_idx])
    np.random.shuffle(balanced_idx)

    X_balanced = torch.tensor(X_seq[balanced_idx], dtype=torch.float32)
    y_balanced = torch.tensor(y_seq[balanced_idx], dtype=torch.long)

    print(f"[✔] Final Sequence Dataset Shape: {X_balanced.shape}")
    torch.save({"X": X_balanced, "y": y_balanced}, "data/unsw_sequences.pt")
    print("[✔] Saved sequence tensors to data/unsw_sequences.pt")

if __name__ == "__main__":
    process_unsw_dataset()