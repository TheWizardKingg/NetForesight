from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
import os
import threading
import time
from collections import Counter, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
import psutil

try:
    import pyshark
except ImportError:
    pyshark = None

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------------
# 1. Model Definitions & Artifact Loaders
# ---------------------------------------------------------
FEATURE_NAMES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt"
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

class AttackForecasterTransformer(nn.Module):
    def __init__(self, feature_dim=16, seq_len=5, num_classes=3, d_model=64, nhead=4, num_layers=2):
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

# Loaded globally during startup
scaler = None
label_encoder = None
xgboost_model = None
shap_explainer = None
transformer_model = None
class_names = []

# ---------------------------------------------------------
# 2. Shared Capture State & Sliding Window Buffer
# ---------------------------------------------------------
capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": Counter(),
    "incoming": 0,
    "outgoing": 0
}
stats_lock = threading.Lock()

# Maintain a rolling window buffer of the last 5 flow feature vectors (5x16)
flow_sequence_buffer = deque(maxlen=5)

def analyze_sequence_window(window):
    """
    window: NumPy array of shape (5, 16)
    """
    # 1. XGBoost Inference on latest timestamp (1, 16)
    xgb_input = window[-1].reshape(1, -1)
    current_probs = xgboost_model.predict_proba(xgb_input)[0]
    current_class_idx = np.argmax(current_probs)
    current_stage = label_encoder.inverse_transform([current_class_idx])[0]
    confidence = float(np.max(current_probs) * 100)
    
    # 2. PyTorch Transformer Forecast on sequence (1, 5, 16)
    transformer_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        forecast_logits = transformer_model(transformer_input)
        forecast_probs = torch.softmax(forecast_logits, dim=1).numpy()[0]
        forecast_class_idx = np.argmax(forecast_probs)
        predicted_stage = label_encoder.inverse_transform([forecast_class_idx])[0]
        
    # 3. Calculate Risk Score (Dynamic based on attack class probability)
    risk_score = round(float((1.0 - current_probs[0]) * 100), 2) if len(current_probs) > 1 else round(confidence, 2)

    return {
        "current_stage": current_stage,
        "predicted_stage": predicted_stage,
        "confidence": confidence,
        "risk_score": risk_score,
        "current_class": current_stage,       
        "forecast_class": predicted_stage     
    }

def get_local_fallback():
    counters = psutil.net_io_counters()
    connections = psutil.net_connections(kind="inet")
    active = sum(
        1 for c in connections
        if c.status in {"ESTABLISHED", "SYN_SENT", "SYN_RECV"}
    )
    return {
        "packets": counters.packets_sent + counters.packets_recv,
        "bytes": counters.bytes_sent + counters.bytes_recv,
        "incoming": counters.packets_recv,
        "outgoing": counters.packets_sent,
        "flows": active,
        "protocols": {}
    }

def packet_capture_worker():
    if pyshark is None:
        return

    # Create and set a fresh asyncio event loop for this background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    interface = os.getenv("NETFORESIGHT_INTERFACE") or None

    try:
        capture = pyshark.LiveCapture(interface=interface)

        for packet in capture.sniff_continuously():
            try:
                length = int(getattr(packet, "length", 0) or 0)
                src = getattr(packet.ip, "src", "") if hasattr(packet, "ip") else ""
                dst = getattr(packet.ip, "dst", "") if hasattr(packet, "ip") else ""
                protocol = getattr(packet, "highest_layer", "UNKNOWN") or "UNKNOWN"

                src_port = 0
                dst_port = 0
                syn_flag = 0
                rst_flag = 0
                ack_flag = 0

                if hasattr(packet, "tcp"):
                    src_port = int(getattr(packet.tcp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.tcp, "dstport", 0) or 0)
                    flags = str(getattr(packet.tcp, "flags", ""))
                    if "0x0002" in flags or "S" in flags:
                        syn_flag = 1
                    if "0x0004" in flags or "R" in flags:
                        rst_flag = 1
                    if "0x0010" in flags or "A" in flags:
                        ack_flag = 1
                elif hasattr(packet, "udp"):
                    src_port = int(getattr(packet.udp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.udp, "dstport", 0) or 0)

                proto_num = 6 if protocol == "TCP" else (17 if protocol == "UDP" else 0)
                flow = (src, dst, src_port, dst_port, protocol)

                feature_vector = [
                    float(dst_port),
                    float(proto_num),
                    0.05,
                    1.0,
                    0.0,
                    float(length),
                    0.0,
                    float(length),
                    float(length),
                    float(length / 0.05),
                    float(1.0 / 0.05),
                    0.01,
                    0.001,
                    float(syn_flag),
                    float(rst_flag),
                    float(ack_flag)
                ]

                with stats_lock:
                    capture_stats["packets"] += 1
                    capture_stats["bytes"] += length
                    capture_stats["flows"].add(flow)
                    capture_stats["protocols"][protocol] += 1
                    flow_sequence_buffer.append(feature_vector)

            except Exception:
                continue

    except Exception as exc:
        print(f"Live packet capture unavailable: {exc}")
    finally:
        loop.close()
        
def get_capture_snapshot():
    with stats_lock:
        return {
            "packets": capture_stats["packets"],
            "bytes": capture_stats["bytes"],
            "flows": len(capture_stats["flows"]),
            "protocols": dict(capture_stats["protocols"]),
            "buffer": list(flow_sequence_buffer)
        }

@app.get("/")
async def root():
    return {
        "status": "NETFORESIGHT backend running with ML Models loaded",
        "websocket": "/ws/alerts",
        "traffic_source": "pyshark_live_capture" if pyshark else "psutil_fallback"
    }

def build_network_update(node_id, window, packets=0, bytes=0):
    window_arr = np.array(window)
    
    # Transformer expects exactly 5 vectors. Don't crash if the buffer is still warming up.
    if window_arr.ndim < 2 or len(window_arr) < 5:
        return {
            "node_id": node_id,
            "packets_delta": packets,
            "bytes_delta": bytes,
            "risk_score": 0.0,
            "current_stage": "Initializing...",
            "predicted_stage": "Initializing...",
            "confidence": 0.0,
            "details": {}
        }
        
    inference = analyze_sequence_window(window_arr)
    
    return {
        "node_id": node_id,
        "packets_delta": packets,
        "bytes_delta": bytes,
        "risk_score": inference.get("risk_score", 0.0),
        "current_stage": inference.get("current_stage", "Unknown"),
        "predicted_stage": inference.get("predicted_stage", "Unknown"),
        "confidence": inference.get("confidence", 0.0),
        "details": inference
    }

@app.get("/api/alerts")
async def alerts():
    snapshot = get_capture_snapshot()
    return build_network_update("REST-API", snapshot.get("buffer", []), 0, 0)

class ConnectionManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, data):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                dead.append(connection)

        for connection in dead:
            self.disconnect(connection)

manager = ConnectionManager()

async def generate_live_data():
    previous_packets = 0
    previous_bytes = 0

    while True:
        await asyncio.sleep(1)

        snapshot = get_capture_snapshot()
        buffer_data = snapshot.get("buffer", [])

        if pyshark and snapshot["packets"] > 0:
            packets_delta = snapshot["packets"] - previous_packets
            bytes_delta = snapshot["bytes"] - previous_bytes
            previous_packets = snapshot["packets"]
            previous_bytes = snapshot["bytes"]
        else:
            live = get_local_fallback()
            packets_delta = live["packets"] - previous_packets
            bytes_delta = live["bytes"] - previous_bytes
            previous_packets = live["packets"]
            previous_bytes = live["bytes"]

        # Actually pass the buffer data to the function, not the packet delta...
        data = build_network_update(
            node_id="LIVE-WS",
            window=buffer_data,
            packets=max(0, packets_delta),
            bytes=max(0, bytes_delta)
        )

        if manager.active_connections:
            await manager.broadcast(data)

@app.on_event("startup")
async def startup_event():
    global scaler, label_encoder, xgboost_model, shap_explainer, transformer_model, class_names

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        def get_path(filename):
            return os.path.join(current_dir, filename)

        print(f"[*] Hunting for ML models in: {current_dir}")
        
        scaler = joblib.load(get_path("feature_scaler.pkl"))
        label_encoder = joblib.load(get_path("label_encoder.pkl"))
        xgboost_model = joblib.load(get_path("xgboost_model.pkl"))
        
        try:
            shap_explainer = joblib.load(get_path("shap_explainer.pkl"))
        except FileNotFoundError:
            shap_explainer = None

        class_names = label_encoder.classes_
        num_classes = len(class_names)

        transformer_model = AttackForecasterTransformer(
            feature_dim=len(FEATURE_NAMES), 
            seq_len=5, 
            num_classes=num_classes
        )
        
        transformer_model.load_state_dict(
            torch.load(get_path("transformer_forecaster.pt"), map_location=torch.device('cpu'))
        )
        transformer_model.eval()

        print("[+] NETFORESIGHT: All ML Models & Explainers successfully integrated!")
    except Exception as exc:
        print(f"[-] FATAL: Failed to load ML artifacts: {exc}")

    if pyshark is not None:
        threading.Thread(target=packet_capture_worker, daemon=True).start()
    asyncio.create_task(generate_live_data())

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)