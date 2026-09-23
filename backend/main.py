import os
import sys
import time
import asyncio
import threading
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import joblib

try:
    import pyshark
except ImportError:
    pyshark = None

#FastAPI App
app = FastAPI(title="NetForesight Backend API", version="2.0")

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Thread-Safe Memory Buffers
stats_lock = threading.Lock()
capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": defaultdict(int)
}

# 5-step sliding window buffer for sequence forecasting
flow_sequence_buffer = deque(maxlen=5)

# Artifact file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# Model & Preprocessing Storage
label_encoder = None
feature_scaler = None
feature_names = None
xgboost_model = None
transformer_model = None


# ---------------------------------------------------------
# PyTorch Transformer Forecaster Architecture
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Application Lifecycle & Model Loading
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global label_encoder, feature_scaler, feature_names, xgboost_model, transformer_model

    print("[+] Initializing NetForesight Backend Services...")

    # 1. Load Preprocessing Artifacts
    try:
        label_encoder = joblib.load(get_path("label_encoder.pkl"))
        feature_scaler = joblib.load(get_path("feature_scaler.pkl"))
        feature_names = joblib.load(get_path("feature_names.pkl"))
        print(f"[✔] Artifacts Loaded: {len(feature_names)} Features | {len(label_encoder.classes_)} Classes")
    except Exception as exc:
        print(f"[-] Critical Error: Failed to load preprocessing artifacts: {exc}")
        print("[-] Ensure you ran `python build_unsw_sequences.py` first!")
        return

    # 2. Load XGBoost Classifier
    try:
        xgboost_model = joblib.load(get_path("xgboost_model.pkl"))
        print("[✔] XGBoost Model loaded successfully.")
    except Exception as exc:
        print(f"[!] Warning: Could not load XGBoost model: {exc}")

    # 3. Load PyTorch Transformer Forecaster
    try:
        num_classes = len(label_encoder.classes_)
        feature_dim = len(feature_names)

        transformer_model = AttackForecasterTransformer(
            feature_dim=feature_dim,
            seq_len=5,
            num_classes=num_classes
        )
        model_path = get_path("transformer_forecaster.pt")
        if os.path.exists(model_path):
            transformer_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            transformer_model.eval()
            print("[✔] Transformer Forecaster loaded successfully.")
        else:
            print("[!] Warning: `transformer_forecaster.pt` not found.")
    except Exception as exc:
        print(f"[-] Critical Error loading Transformer model: {exc}")

    # 4. Start Background Live Packet Capture Thread
    capture_thread = threading.Thread(target=packet_capture_worker, daemon=True)
    capture_thread.start()
    print("[✔] Packet capture worker thread initiated.")


# ---------------------------------------------------------
# Live Packet Capture Worker Thread
# ---------------------------------------------------------
def packet_capture_worker():
    if pyshark is None:
        print("[-] PyShark module not installed. Live packet sniffing is disabled.")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Use network adapter index string obtained from `tshark -D`
    interface = "4"
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"

    try:
        capture = pyshark.LiveCapture(interface=interface, tshark_path=tshark_path)

        for packet in capture.sniff_continuously():
            try:
                length = int(getattr(packet, "length", 0) or 0)
                src = getattr(packet.ip, "src", "") if hasattr(packet, "ip") else ""
                dst = getattr(packet.ip, "dst", "") if hasattr(packet, "ip") else ""
                protocol = getattr(packet, "highest_layer", "UNKNOWN") or "UNKNOWN"

                src_port = 0
                dst_port = 0
                if hasattr(packet, "tcp"):
                    src_port = int(getattr(packet.tcp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.tcp, "dstport", 0) or 0)
                elif hasattr(packet, "udp"):
                    src_port = int(getattr(packet.udp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.udp, "dstport", 0) or 0)

                flow = (src, dst, src_port, dst_port, protocol)

                # Construct feature vector matching UNSW feature dimensions
                num_feats = len(feature_names) if feature_names else 42
                feature_vector = [0.0] * num_feats

                # Map primary live packet properties to corresponding indices
                feature_vector[0] = float(length)
                feature_vector[1] = float(src_port)
                feature_vector[2] = float(dst_port)

                with stats_lock:
                    capture_stats["packets"] += 1
                    capture_stats["bytes"] += length
                    capture_stats["flows"].add(flow)
                    capture_stats["protocols"][protocol] += 1
                    flow_sequence_buffer.append(feature_vector)

            except Exception:
                continue

    except Exception as exc:
        print(f"[-] Live packet capture violently crashed: {exc}")
    finally:
        loop.close()


# ---------------------------------------------------------
# WebSocket Connection Manager for Live UI Alerts
# ---------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            with stats_lock:
                current_packets = capture_stats["packets"]
                current_bytes = capture_stats["bytes"]
                current_flows = len(capture_stats["flows"])

            await websocket.send_json({
                "type": "telemetry",
                "status": "online",
                "packets": current_packets,
                "bytes": current_bytes,
                "flows": current_flows
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "service": "NetForesight API"}


@app.get("/api/capture/stats")
def get_capture_stats():
    with stats_lock:
        return {
            "packets": capture_stats["packets"],
            "bytes": capture_stats["bytes"],
            "flows": len(capture_stats["flows"]),
            "protocols": dict(capture_stats["protocols"]),
            "buffer_size": len(flow_sequence_buffer)
        }


@app.get("/api/predict/forecast")
def forecast_next_stage():
    if transformer_model is None or label_encoder is None or feature_scaler is None:
        raise HTTPException(status_code=500, detail="Models or preprocessing artifacts are uninitialized.")

    with stats_lock:
        current_buffer = list(flow_sequence_buffer)

    num_feats = len(feature_names)
    # Fill buffer with dummy zeros if live traffic hasn't reached 5 packets yet
    while len(current_buffer) < 5:
        current_buffer.insert(0, [0.0] * num_feats)

    try:
        # Scale incoming raw sequence
        scaled_seq = feature_scaler.transform(np.array(current_buffer))
        tensor_input = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 5, num_features)

        with torch.no_grad():
            outputs = transformer_model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]

        top_class_idx = int(np.argmax(probabilities))
        predicted_class = label_encoder.inverse_transform([top_class_idx])[0]

        class_probabilities = {
            cls_name: float(prob)
            for cls_name, prob in zip(label_encoder.classes_, probabilities)
        }

        return {
            "predicted_next_stage": predicted_class,
            "confidence": float(probabilities[top_class_idx]),
            "class_probabilities": class_probabilities
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)