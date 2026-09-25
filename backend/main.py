import os
import sys
import time
import asyncio
import multiprocessing as mp
from queue import Empty, Full
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pyshark

# ---------------------------------------------------------
# PyShark Background Sniffer Worker Process
# ---------------------------------------------------------
def pyshark_worker_process(packet_queue, interface="4", tshark_path=None):
    """
    Dedicated OS process for PyShark to prevent tshark stdout parsing 
    from freezing FastAPI's GIL and event loop.
    """
    try:
        kwargs = {
            "interface": interface,
            "bpf_filter": "ip and (tcp or udp)",
            "use_json": True,
            "include_raw": False
        }
        if tshark_path and os.path.exists(tshark_path):
            kwargs["tshark_path"] = tshark_path

        capture = pyshark.LiveCapture(**kwargs)

        for packet in capture.sniff_continuously():
            try:
                length = int(getattr(packet, "length", 0) or 0)
                src_port = 0
                dst_port = 0
                is_tcp = 0.0
                protocol = "OTHER"

                if hasattr(packet, "tcp"):
                    src_port = int(getattr(packet.tcp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.tcp, "dstport", 0) or 0)
                    is_tcp = 1.0
                    protocol = "TCP"
                elif hasattr(packet, "udp"):
                    src_port = int(getattr(packet.udp, "srcport", 0) or 0)
                    dst_port = int(getattr(packet.udp, "dstport", 0) or 0)
                    protocol = "UDP"

                pkt_tuple = (time.time(), length, src_port, dst_port, is_tcp, protocol)

                # Non-blocking enqueue: drop packet if queue buffer is full to prevent high latency
                try:
                    packet_queue.put_nowait(pkt_tuple)
                except Full:
                    pass

            except Exception:
                continue

    except Exception as exc:
        print(f"[-] PyShark worker crashed: {exc}")


# FastAPI App Setup
app = FastAPI(title="NetForesight PyShark Backend", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

label_encoder = None
feature_scaler = None
feature_names = None
xgboost_model = None
transformer_model = None

# Multiprocessing Inter-Process Queue and Process Handle
packet_queue = mp.Queue(maxsize=5000)
pyshark_process_handle = None

# Telemetry Counter Storage
capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": defaultdict(int)
}


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
# Sliding Window Aggregator
# ---------------------------------------------------------
class RealtimeFlowAggregator:
    def __init__(self, window_sec=2.0, seq_len=5):
        self.window_sec = window_sec
        self.seq_len = seq_len
        self.packet_history = deque()
        self.flow_sequence = deque(maxlen=seq_len)
        self.smoothed_prob = 0.0
        self.alpha = 0.3

    def add_packet(self, pkt_time, length, src_port, dst_port, is_tcp):
        self.packet_history.append((pkt_time, length, src_port, dst_port, is_tcp))

    def update_window_and_extract_features(self, num_feats=42):
        now = time.time()
        cutoff = now - self.window_sec

        while self.packet_history and self.packet_history[0][0] < cutoff:
            self.packet_history.popleft()

        pkts = list(self.packet_history)
        pkt_count = len(pkts)

        feat_vec = [0.0] * num_feats

        if pkt_count > 0:
            total_bytes = sum(p[1] for p in pkts)
            dur = max(now - pkts[0][0], 0.001)
            sload = (total_bytes * 8.0) / dur
            rate = pkt_count / dur
            avg_pkt_size = total_bytes / float(pkt_count)
            tcp_ratio = sum(1 for p in pkts if p[4]) / float(pkt_count)

            feat_vec[0] = float(dur)
            feat_vec[1] = float(pkts[-1][2])
            feat_vec[2] = float(pkts[-1][3])
            feat_vec[3] = float(total_bytes)
            feat_vec[4] = float(sload)
            feat_vec[5] = float(rate)
            feat_vec[6] = float(avg_pkt_size)
            feat_vec[7] = float(pkt_count)
            feat_vec[8] = float(tcp_ratio)

        self.flow_sequence.append(feat_vec)

        seq_array = list(self.flow_sequence)
        while len(seq_array) < self.seq_len:
            seq_array.insert(0, [0.0] * num_feats)

        return np.array(seq_array)

    def smooth_prediction(self, raw_prob):
        self.smoothed_prob = (self.alpha * raw_prob) + ((1.0 - self.alpha) * self.smoothed_prob)
        return self.smoothed_prob


aggregator = RealtimeFlowAggregator(window_sec=2.0, seq_len=5)


def drain_pyshark_queue():
    """Drains pending packets from PyShark IPC queue into local stats and flow aggregator."""
    while True:
        try:
            pkt_time, length, src_port, dst_port, is_tcp, protocol = packet_queue.get_nowait()
            capture_stats["packets"] += 1
            capture_stats["bytes"] += length
            capture_stats["protocols"][protocol] += 1
            aggregator.add_packet(pkt_time, length, src_port, dst_port, is_tcp)
        except Empty:
            break


# ---------------------------------------------------------
# FastAPI Lifecycle
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global label_encoder, feature_scaler, feature_names, xgboost_model, transformer_model, pyshark_process_handle

    print("[+] Initializing NetForesight PyShark Backend...")

    # Load artifacts
    try:
        label_encoder = joblib.load(get_path("label_encoder.pkl"))
        feature_scaler = joblib.load(get_path("feature_scaler.pkl"))
        feature_names = joblib.load(get_path("feature_names.pkl"))
        print(f"[✔] Artifacts Loaded: {len(feature_names)} Features | {len(label_encoder.classes_)} Classes")
    except Exception as exc:
        print(f"[-] Critical Error loading preprocessing artifacts: {exc}")
        return

    # Load XGBoost
    try:
        xgboost_model = joblib.load(get_path("xgboost_model.pkl"))
        print("[✔] XGBoost Model loaded.")
    except Exception as exc:
        print(f"[!] Warning: Could not load XGBoost model: {exc}")

    # Load PyTorch Transformer
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

    # Start PyShark Multiprocessing Worker Process
    try:
        tshark_win_path = r"C:\Program Files\Wireshark\tshark.exe"
        pyshark_process_handle = mp.Process(
            target=pyshark_worker_process,
            args=(packet_queue, "4", tshark_win_path),
            daemon=True
        )
        pyshark_process_handle.start()
        print("[✔] PyShark isolated worker process spawned.")
    except Exception as exc:
        print(f"[-] Failed to launch PyShark worker process: {exc}")


@app.on_event("shutdown")
def shutdown_event():
    global pyshark_process_handle
    if pyshark_process_handle and pyshark_process_handle.is_alive():
        pyshark_process_handle.terminate()
        pyshark_process_handle.join()
        print("[+] PyShark background process terminated.")


# ---------------------------------------------------------
# WebSocket Connection Manager
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
            drain_pyshark_queue()
            await websocket.send_json({
                "type": "telemetry",
                "status": "online",
                "packets": capture_stats["packets"],
                "bytes": capture_stats["bytes"],
                "flows": len(capture_stats["flows"])
            })
            await asyncio.sleep(1)
    except (WebSocketDisconnect, Exception):
        manager.disconnect(websocket)


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "service": "NetForesight API v3.1 (PyShark Isolated)"}


@app.get("/api/capture/stats")
def get_capture_stats():
    drain_pyshark_queue()
    return {
        "packets": capture_stats["packets"],
        "bytes": capture_stats["bytes"],
        "flows": len(capture_stats["flows"]),
        "protocols": dict(capture_stats["protocols"]),
        "buffer_size": len(aggregator.packet_history)
    }


@app.get("/api/predict/forecast")
def forecast_next_stage():
    if transformer_model is None or label_encoder is None or feature_scaler is None:
        raise HTTPException(status_code=500, detail="Models or preprocessing artifacts are uninitialized.")

    # Drain packet queue before running window feature extraction
    drain_pyshark_queue()

    num_feats = len(feature_names) if feature_names else 42
    raw_sequence = aggregator.update_window_and_extract_features(num_feats=num_feats)

    try:
        scaled_seq = feature_scaler.transform(raw_sequence)
        tensor_input = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = transformer_model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]

        top_class_idx = int(np.argmax(probabilities))
        raw_confidence = float(probabilities[top_class_idx])

        smoothed_confidence = aggregator.smooth_prediction(raw_confidence)
        predicted_class = label_encoder.inverse_transform([top_class_idx])[0]

        class_probabilities = {
            cls_name: float(prob)
            for cls_name, prob in zip(label_encoder.classes_, probabilities)
        }

        return {
            "predicted_next_stage": predicted_class,
            "confidence": float(smoothed_confidence),
            "raw_confidence": float(raw_confidence),
            "class_probabilities": class_probabilities,
            "window_packet_count": len(aggregator.packet_history)
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    # Required for Windows multiprocessing compatibility
    mp.freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)