from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
import psutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "status": "NETFORESIGHT backend running",
        "websocket": "/ws/alerts",
        "traffic_source": "local_machine"
    }

def get_live_traffic():
    counters = psutil.net_io_counters()
    connections = psutil.net_connections(kind="inet")

    active_connections = sum(
        1 for connection in connections
        if connection.status in {"ESTABLISHED", "SYN_SENT", "SYN_RECV"}
    )

    return {
        "bytes_sent": counters.bytes_sent,
        "bytes_recv": counters.bytes_recv,
        "packets_sent": counters.packets_sent,
        "packets_recv": counters.packets_recv,
        "connections": active_connections
    }

@app.get("/api/alerts")
async def alerts():
    traffic = get_live_traffic()

    return {
        "status": "ok",
        "source": "local_machine",
        "risk": 0,
        "next_attack": "Analyzing",
        "mitre": "—",
        "confidence": 0,
        "flows": traffic["connections"],
        "packets": traffic["packets_sent"] + traffic["packets_recv"],
        "anomaly": 0
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
    previous = get_live_traffic()

    while True:
        await asyncio.sleep(1)

        current = get_live_traffic()

        packets_delta = (
            current["packets_sent"] + current["packets_recv"]
            - previous["packets_sent"] - previous["packets_recv"]
        )

        bytes_delta = (
            current["bytes_sent"] + current["bytes_recv"]
            - previous["bytes_sent"] - previous["bytes_recv"]
        )

        outgoing_packets = current["packets_sent"] - previous["packets_sent"]
        incoming_packets = current["packets_recv"] - previous["packets_recv"]

        packets_per_second = max(0, packets_delta)
        flows_per_second = max(0, current["connections"])

        data = {
            "type": "network_update",
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": "Live network traffic",
            "risk": 0,
            "next_attack": "Analyzing",
            "mitre": "—",
            "confidence": 0,
            "flows": flows_per_second,
            "packets": packets_per_second,
            "anomaly": 0,
            "status": "MONITORING",
            "bytes_per_second": max(0, bytes_delta),
            "incoming_packets": max(0, incoming_packets),
            "outgoing_packets": max(0, outgoing_packets),
            "active_connections": current["connections"],
            "source": "local_machine"
        }

        if manager.active_connections:
            await manager.broadcast(data)

        previous = current

@app.on_event("startup")
async def startup_event():
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
