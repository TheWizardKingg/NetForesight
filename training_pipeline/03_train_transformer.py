import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

print("[*] Training PyTorch Transformer...")

# 1. LOAD PREPROCESSED DATA & METADATA
X_seq_train = np.load("training_pipeline/tmp/X_seq_train.npy")
y_seq_train = np.load("training_pipeline/tmp/y_seq_train.npy")
label_encoder = joblib.load("backend/label_encoder.pkl")

train_dataset = TensorDataset(
    torch.tensor(X_seq_train, dtype=torch.float32), 
    torch.tensor(y_seq_train, dtype=torch.long)
)

# num_workers=0 avoids Windows MINGW multiprocessing crashes
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0) 

# 2. TRANSFORMER ARCHITECTURE (Matching main.py exactly)
class AttackForecasterTransformer(nn.Module):
    def __init__(self, feature_dim=16, seq_len=5, num_classes=len(label_encoder.classes_), d_model=64, nhead=4, num_layers=2):
        super(AttackForecasterTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# 3. INITIALIZE MODEL, OPTIMIZER & SCHEDULER
device = torch.device("cpu")
transformer_model = AttackForecasterTransformer().to(device)

EPOCHS = 25
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"[*] Starting training for {EPOCHS} epochs overnight...")
transformer_model.train()

# 4. OVERNIGHT TRAINING LOOP
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = transformer_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"    Epoch {epoch+1:02d}/{EPOCHS} - Loss: {avg_loss:.5f} - LR: {current_lr:.6f}")

# 5. DEPLOY WEIGHTS DIRECTLY TO BACKEND
torch.save(transformer_model.state_dict(), "backend/transformer_forecaster.pt")
print("[+] Transformer Complete. Model trained and deployed to backend/transformer_forecaster.pt.")