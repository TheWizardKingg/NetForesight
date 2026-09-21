from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
import os
import threading
import time
from collections import Counter

import psutil

try:
    import pyshark
except ImportError:
    pyshark = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

capture_stats = {
    "packets": 0,
    "bytes": 0,
    "flows": set(),
    "protocols": Counter(),
    "incoming": 0,
    "outgoing": 0
}
stats_lock = threading.Lock()

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

    interface = os.getenv("NETFORESIGHT_INTERFACE") or None

    try:
        capture = pyshark.LiveCapture(interface=interface)

        for packet in capture.sniff_continuously():
            try:
                length = int(getattr(packet, "length", 0) or 0)
                src = getattr(packet.ip, "src", "") if hasattr(packet, "ip") else ""
                dst = getattr(packet.ip, "dst", "") if hasattr(packet, "ip") else ""
                protocol = getattr(packet, "highest_layer", "UNKNOWN") or "UNKNOWN"

                src_port = ""
                dst_port = ""

                if hasattr(packet, "tcp"):
                    src_port = getattr(packet.tcp, "srcport", "")
                    dst_port = getattr(packet.tcp, "dstport", "")
                elif hasattr(packet, "udp"):
                    src_port = getattr(packet.udp, "srcport", "")
                    dst_port = getattr(packet.udp, "dstport", "")

                flow = (src, dst, src_port, dst_port, protocol)

                with stats_lock:
                    capture_stats["packets"] += 1
                    capture_stats["bytes"] += length
                    capture_stats["flows"].add(flow)
                    capture_stats["protocols"][protocol] += 1

            except Exception:
                continue

    except Exception as exc:
        print(f"Live packet capture unavailable: {exc}")

def get_capture_snapshot():
    with stats_lock:
        return {
            "packets": capture_stats["packets"],
            "bytes": capture_stats["bytes"],
            "flows": len(capture_stats["flows"]),
            "protocols": dict(capture_stats["protocols"])
        }

@app.get("/")
async def root():
    return {
        "status": "NETFORESIGHT backend running",
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

    anomaly = min(10.0, max(0.0, packets / 1000.0))
    risk = min(100, round(anomaly * 10))

    return {
        "type": "network_update",
        "time": datetime.now().strftime("%H:%M:%S"),
        "event": "Live network traffic",
        "risk": risk,
        "next_attack": "Analyzing",
        "mitre": "—",
        "confidence": 0,
        "flows": flows,
        "packets": packets,
        "anomaly": anomaly,
        "status": "MONITORING",
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
