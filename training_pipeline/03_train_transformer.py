import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

print("--- BOOTING TRANSFORMER TRAINING ---")
X_seq = np.load('training_pipeline/X_seq.npy')
y_next = np.load('training_pipeline/y_next.npy')
encoder = joblib.load('training_pipeline/label_encoder.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_next, test_size=0.2, random_state=42)

# FIXING THE CLASS IMBALANCE: Brutally weight the minority attack classes
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print(f"Applied Class Weights to stop Benign spam: {weights}")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=256, shuffle=True)

class NextStageTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Grab the output of the last sequence step
        out = self.fc(x[:, -1, :])
        return out

input_dim = X_seq.shape[2]
num_classes = len(encoder.classes_)

model = NextStageTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights) # <--- THIS FIXES THE LAZY BENIGN PREDICTIONS
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epochs = 15
print("Training Transformer...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'transformer_forecaster.pt')
print("Saved transformer_forecaster.pt. You're done.")