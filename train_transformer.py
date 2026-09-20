import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

print("[+] Phase 3 Started: Training Temporal Transformer (10 Epochs)...")

# 1. Load sequence binaries generated from Phase 1
X_seq = np.load("X_sequences.npy")  # Shape: (N, 5, 16)
y_seq = np.load("y_targets.npy")    # Shape: (N,)

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")
class_names = label_encoder.classes_
num_classes = len(class_names)

# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# PyTorch Dataset Definition
class NetworkSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NetworkSequenceDataset(X_train, y_train)
val_dataset = NetworkSequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# 2. Transformer Architecture for Temporal Next-Stage Forecasting
class AttackForecasterTransformer(nn.Module):
    def __init__(self, feature_dim, seq_len, num_classes, d_model=64, nhead=4, num_layers=2):
        super(AttackForecasterTransformer, self).__init__()
        
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            batch_first=True
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
        out = self.fc_out(x)
        return out

seq_len = X_seq.shape[1]
feature_dim = X_seq.shape[2]

model = AttackForecasterTransformer(feature_dim=feature_dim, seq_len=seq_len, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 3. CPU-Optimized Training Loop (10 Epochs)
EPOCHS = 10
device = torch.device("cpu")
model.to(device)

print(f"[+] Architecture loaded. Training on CPU for {EPOCHS} epochs...")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    scheduler.step()
    train_acc = (correct / total) * 100
    avg_loss = total_loss / total
    
    # Validation Loop
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
            
    val_acc = (val_correct / val_total) * 100
    print(f"    Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Save checkpoint if best validation score
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "transformer_forecaster.pt")

print(f"\n[+] Phase 3 Complete! Best Model saved as 'transformer_forecaster.pt' with Val Accuracy: {best_val_acc:.2f}%")

# 4. Final Evaluation Report
print("\n[+] Generating Final Validation Metrics...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_targets.extend(y_batch.numpy())

print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))