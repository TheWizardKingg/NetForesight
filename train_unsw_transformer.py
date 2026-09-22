import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

class AttackForecasterTransformer(nn.Module):
    def __init__(self, feature_dim, seq_len=5, num_classes=10, d_model=64, nhead=4, num_layers=2):
        super().__init__()
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

def train():
    data = torch.load("data/unsw_sequences.pt")
    X, y = data["X"], data["y"]
    le = joblib.load("backend/label_encoder.pkl")
    num_classes = len(le.classes_)
    feature_dim = X.shape[2]

    print(f"[*] Training Transformer: Sequences={X.shape[0]} | Features={feature_dim} | Classes={num_classes}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = AttackForecasterTransformer(feature_dim=feature_dim, seq_len=5, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(loader):.4f} | Accuracy: {(correct/total)*100:.2f}%")

    torch.save(model.state_dict(), "backend/transformer_forecaster.pt")
    print("[✔] Saved backend/transformer_forecaster.pt")

if __name__ == "__main__":
    train()