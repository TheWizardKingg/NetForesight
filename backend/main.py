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
        self.input_projection = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
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
        return self.fc_out(x)

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

def analyze_sequence_window(sequence_window_raw):
    """
    Unified Inference Engine combining XGBoost current state classification,
    PyTorch Transformer next-stage forecasting, and SHAP triggers.
    """
    if scaler is None or xgboost_model is None or transformer_model is None:
        return None

    df_raw = pd.DataFrame(sequence_window_raw, columns=FEATURE_NAMES)
    sequence_scaled = scaler.transform(df_raw)
    
    # --- A. XGBoost Classification ---
    xgb_input = sequence_scaled.reshape(1, -1)
    current_class_idx = xgboost_model.predict(xgb_input)[0]
    current_class_label = class_names[current_class_idx]
    current_probs = xgboost_model.predict_proba(xgb_input)[0]
    current_confidence = float(np.max(current_probs))
    
    # --- B. PyTorch Transformer Forecast ---
    tensor_input = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = transformer_model(tensor_input)
        forecast_probs = torch.softmax(logits, dim=1).numpy()[0]
        
    forecast_class_idx = np.argmax(forecast_probs)
    forecast_class_label = class_names[forecast_class_idx]
    forecast_confidence = float(forecast_probs[forecast_class_idx])
    
    # --- C. SHAP Feature Attribution ---
    try:
        shap_vals = shap_explainer(xgb_input)
        if len(shap_vals.values.shape) == 3:
            feature_importance = np.abs(shap_vals.values[0, :, current_class_idx])
        else:
            feature_importance = np.abs(shap_vals.values[0])
            
        feature_importance_reshaped = feature_importance.reshape(5, 16).mean(axis=0)
        top_feature_indices = np.argsort(feature_importance_reshaped)[::-1][:3]
        
        top_triggers = [
            {
                "feature": FEATURE_NAMES[idx],
                "value": round(float(sequence_window_raw[-1][idx]), 2),
                "importance_score": float(feature_importance_reshaped[idx])
            }
            for idx in top_feature_indices
        ]
    except Exception:
        top_triggers = []

    # --- D. Dynamic Risk Score Calculation ---
    base_risk = 10 if current_class_label == "Benign" else 70
    threat_multiplier = 1.0 if current_class_label == "Benign" else 1.3
    risk_score = min(100, int((base_risk * current_confidence * threat_multiplier) + (forecast_confidence * 15)))

    mitre_stage = MITRE_MAP.get(current_class_label, "Reconnaissance / Normal")
    
    return {
        "predicted_attack": current_class_label,
        "confidence": round(current_confidence * 100, 2),
        "predicted_next_attack": forecast_class_label,
        "forecast_confidence": round(forecast_confidence * 100, 2),
        "risk_score": risk_score,
        "mitre_stage": mitre_stage,
        "top_triggers": top_triggers
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

@app.get("/api/alerts")
async def alerts():
    return build_network_update(0, 0)

def build_network_update(packets_per_second, bytes_per_second):
    snapshot = get_capture_snapshot()

    if pyshark and snapshot["packets"] > 0:
        flows = snapshot["flows"]
        packets = packets_per_second
        incoming = 0
        outgoing = 0
        source = "pyshark_live_capture"
        protocol_counts = snapshot["protocols"]
    else:
        live = get_local_fallback()
        flows = live["flows"]
        packets = packets_per_second or 0
        incoming = live["incoming"]
        outgoing = live["outgoing"]
        source = "psutil_fallback"
        protocol_counts = live["protocols"]

    # Fill sequence window with dummy/zero sequences if less than 5 packets captured yet
    buffer = snapshot["buffer"]
    if len(buffer) < 5:
        pad_size = 5 - len(buffer)
        padded_buffer = [[0.0] * 16] * pad_size + buffer
    else:
        padded_buffer = buffer[-5:]

    raw_window = np.array(padded_buffer, dtype=np.float32)
    inference = analyze_sequence_window(raw_window)

    if inference:
        risk = inference["risk_score"]
        status = "CRITICAL" if risk > 75 else ("ELEVATED" if risk > 45 else "MONITORING")
        event = f"Detected: {inference['predicted_attack']}"
        next_attack = inference["predicted_next_attack"]
        mitre = inference["mitre_stage"]
        confidence = inference["confidence"]
        top_triggers = inference.get("top_triggers", [])
    else:
        anomaly = min(10.0, max(0.0, packets / 1000.0))
        risk = min(100, round(anomaly * 10))
        status = "MONITORING"
        event = "Live network traffic"
        next_attack = "Analyzing"
        mitre = "—"
        confidence = 0
        top_triggers = []

    return {
        "type": "network_update",
        "time": datetime.now().strftime("%H:%M:%S"),
        "event": event,
        "risk": risk,
        "next_attack": next_attack,
        "mitre": mitre,
        "confidence": confidence,
        "top_triggers": top_triggers,
        "flows": flows,
        "packets": packets,
        "anomaly": round(risk / 10.0, 1),
        "status": status,
        "bytes_per_second": bytes_per_second,
        "incoming_packets": incoming,
        "outgoing_packets": outgoing,
        "active_connections": flows,
        "protocols": protocol_counts,
        "source": source
    }

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

        data = build_network_update(
            max(0, packets_delta),
            max(0, bytes_delta)
        )

        if manager.active_connections:
            await manager.broadcast(data)

@app.on_event("startup")
async def startup_event():
    global scaler, label_encoder, xgboost_model, shap_explainer, transformer_model, class_names

    # Load ML Artifacts
    try:
        # Resolve project root relative to main.py
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        def get_path(filename):
            path_in_root = os.path.join(base_dir, filename)
            return path_in_root if os.path.exists(path_in_root) else filename

        scaler = joblib.load(get_path("feature_scaler.pkl"))
        label_encoder = joblib.load(get_path("label_encoder.pkl"))
        xgboost_model = joblib.load(get_path("xgboost_model.pkl"))
        shap_explainer = joblib.load(get_path("shap_explainer.pkl"))

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
        print(f"[-] Failed to load ML artifacts: {exc}")

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