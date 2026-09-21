from fastapi import FastAPI, WebSocket
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
    return {"status": "NETFORESIGHT backend running"}

@app.get("/api/alerts")
async def alerts():
    return {
        "risk": 78,
        "next_attack": "Lateral Movement",
        "mitre": "T1021",
        "confidence": 0.78
    }

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()

    events = [
        {
            "event": "Port Scan detected",
            "risk": 52,
            "next_attack": "Credential Access",
            "mitre": "T1046",
            "confidence": 0.71
        },
        {
            "event": "SSH Brute Force detected",
            "risk": 68,
            "next_attack": "Lateral Movement",
            "mitre": "T1110",
            "confidence": 0.79
        },
        {
            "event": "Lateral Movement detected",
            "risk": 78,
            "next_attack": "Command Execution",
            "mitre": "T1021",
            "confidence": 0.84
        },
        {
            "event": "Suspicious Traffic detected",
            "risk": 86,
            "next_attack": "Data Exfiltration",
            "mitre": "T1041",
            "confidence": 0.91
        }
    ]

    index = 0

    while True:
        event = events[index]

        data = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event["event"],
            "risk": event["risk"],
            "next_attack": event["next_attack"],
            "mitre": event["mitre"],
            "confidence": event["confidence"]
        }

        await websocket.send_json(data)

        index = (index + 1) % len(events)

        await asyncio.sleep(5)