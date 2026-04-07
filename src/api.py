"""OTShield FastAPI backend — real-time streaming + dashboard support"""

import asyncio
import random
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from src.fusion import get_risk_score, classify_risk, generate_explanation

# ───────────────────────────────────────────────
# APP SETUP
# ───────────────────────────────────────────────
app = FastAPI(title="OTShield API")

frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)

app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(frontend_path, "dashboard.html"))


# ───────────────────────────────────────────────
# STATE
# ───────────────────────────────────────────────
connected_clients = []
history = deque(maxlen=100)
current_mode = "normal"

# ───────────────────────────────────────────────
# FAKE PLC STREAM (fallback)
# ───────────────────────────────────────────────
def get_current_reading():
    return {
        "L_T1": 5,
        "L_T2": 5,
        "L_T3": 5,
        "FLOW": 30,
        "PRESSURE": 2,
        "ATT_FLAG": 0
    }


# ───────────────────────────────────────────────
# GLOBAL SMOOTH MEMORY
# ───────────────────────────────────────────────
prev_values = {
    "tank_1": 5,
    "tank_2": 5,
    "tank_3": 5,
    "flow": 30,
    "pressure": 2
}


# ───────────────────────────────────────────────
# PAYLOAD BUILDER
# ───────────────────────────────────────────────
def build_payload(reading: dict) -> dict:
    global prev_values

    # ── Scores ──
    cyber_score = round(random.uniform(0.2, 0.9), 4)
    physical_score = round(random.uniform(0.2, 0.9), 4)

    fusion_score, mode = get_risk_score(cyber_score, physical_score)
    risk_level = classify_risk(cyber_score, physical_score, fusion_score)
    explanation = generate_explanation(cyber_score, physical_score)

    # ── Simulated modalities (correlated) ──
    audio_score = min(1.0, max(0.0, physical_score + random.uniform(-0.2, 0.2)))
    visual_score = min(1.0, max(0.0, cyber_score + random.uniform(-0.2, 0.2)))

    # ── Smooth sensor system ──
    def smooth(new_val, prev, step=0.2):
        return round(prev + (new_val - prev) * step, 2)

    target_t1 = reading.get("L_T1", 5) + random.uniform(-0.5, 0.5)
    target_t2 = reading.get("L_T2", 5) + random.uniform(-0.5, 0.5)
    target_t3 = reading.get("L_T3", 5) + random.uniform(-0.5, 0.5)

    target_flow = reading.get("FLOW", 30) + random.uniform(-2, 2)
    target_pressure = reading.get("PRESSURE", 2) + random.uniform(-0.2, 0.2)

    tank_1 = smooth(target_t1, prev_values["tank_1"])
    tank_2 = smooth(target_t2, prev_values["tank_2"])
    tank_3 = smooth(target_t3, prev_values["tank_3"])

    flow = smooth(target_flow, prev_values["flow"])
    pressure = smooth(target_pressure, prev_values["pressure"])

    prev_values.update({
        "tank_1": tank_1,
        "tank_2": tank_2,
        "tank_3": tank_3,
        "flow": flow,
        "pressure": pressure
    })

    # ── SHAP simulation ──
    shap_explanation = {
        "cyber": {
            "pct": int(cyber_score * 100),
            "label": "Network anomaly detected"
        },
        "audio": {
            "pct": int(audio_score * 100),
            "label": "Acoustic baseline normal"
        },
        "visual": {
            "pct": int(visual_score * 100),
            "label": "Visual feed stable"
        }
    }

    # ── Final payload ──
    return {
        "cyber_score": cyber_score * 100,
        "physical_score": physical_score * 100,
        "audio_score": audio_score * 100,
        "visual_score": visual_score * 100,
        "fusion_score": fusion_score * 100,

        "risk_level": risk_level,
        "recommended_action": explanation,

        "mode_active": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "shap_explanation": shap_explanation,

        # Optional sensor data (future UI expansion)
        "sensors": {
            "tank_1": tank_1,
            "tank_2": tank_2,
            "tank_3": tank_3
        },
        "system": {
            "flow_rate": flow,
            "pressure": pressure
        }
    }


# ───────────────────────────────────────────────
# WEBSOCKET STREAM
# ───────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)

    print(f"[WS] Connected | Clients: {len(connected_clients)}")

    try:
        while True:
            try:
                reading = get_current_reading()
                payload = build_payload(reading)
            except Exception as e:
                print("[ERROR]", e)
                continue

            history.append(payload)

            disconnected = []
            for client in connected_clients:
                try:
                    await client.send_json(payload)
                except Exception:
                    disconnected.append(client)

            for client in disconnected:
                if client in connected_clients:
                    connected_clients.remove(client)

            await asyncio.sleep(0.6)  # 🔥 FAST REAL-TIME

    except WebSocketDisconnect:
        print("[WS] Client disconnected")

    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)


# ───────────────────────────────────────────────
# SIMULATION ENDPOINT
# ───────────────────────────────────────────────
@app.post("/simulate/{mode}")
async def simulate(mode: str):
    global current_mode
    current_mode = mode
    return {"status": "ok", "mode": mode}


# ───────────────────────────────────────────────
# STATUS
# ───────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "mode": current_mode,
        "connected_clients": len(connected_clients)
    }