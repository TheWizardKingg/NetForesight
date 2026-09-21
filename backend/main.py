from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime

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
        "websocket": "/ws/alerts"
    }

@app.get("/api/alerts")
async def alerts():
    return {
        "status": "ok",
        "source": "backend",
        "risk": 78,
        "next_attack": "Lateral Movement",
        "mitre": "T1021",
        "confidence": 0.78
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
    events = [
        {
            "event": "Port Scan detected",
            "risk": 52,
            "next_attack": "Credential Access",
            "mitre": "T1046",
            "confidence": 0.71,
            "flows": 94,
            "packets": 2860,
            "anomaly": 3.8,
            "status": "AT RISK"
        },
        {
            "event": "SSH Brute Force detected",
            "risk": 68,
            "next_attack": "Lateral Movement",
            "mitre": "T1110",
            "confidence": 0.79,
            "flows": 121,
            "packets": 4120,
            "anomaly": 6.1,
            "status": "AT RISK"
        },
        {
            "event": "Lateral Movement detected",
            "risk": 78,
            "next_attack": "Command Execution",
            "mitre": "T1021",
            "confidence": 0.84,
            "flows": 154,
            "packets": 5830,
            "anomaly": 7.6,
            "status": "UNDER ATTACK"
        },
        {
            "event": "Suspicious Traffic detected",
            "risk": 86,
            "next_attack": "Data Exfiltration",
            "mitre": "T1041",
            "confidence": 0.91,
            "flows": 187,
            "packets": 7210,
            "anomaly": 8.8,
            "status": "UNDER ATTACK"
        }
    ]

    index = 0

    while True:
        if manager.active_connections:
            event = events[index]

            data = {
                "type": "network_update",
                "time": datetime.now().strftime("%H:%M:%S"),
                **event
            }

            await manager.broadcast(data)
            index = (index + 1) % len(events)

        await asyncio.sleep(5)

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
