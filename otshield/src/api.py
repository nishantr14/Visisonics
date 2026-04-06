"""OTShield FastAPI backend -- WebSocket broadcast + REST endpoints."""

import asyncio
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from fusion import get_risk_score, classify_risk, get_recommended_action
from api_models import PredictionPayload, StatusResponse, ShapExplanation, ShapFeature
from supervised_scorer import get_supervised_score, reset_history

# -- DATA SOURCE SWITCH -----------------------------------------------------
USE_REAL_PLC = False

if USE_REAL_PLC:
    from data_collector import get_current_reading
else:
    from fake_plc_stream import get_current_reading, set_mode as stream_set_mode

# ---------------------------------------------------------------------------

app = FastAPI(title="OTShield API", version="0.2.0")

# -- STATIC FILE SERVING ----------------------------------------------------
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse(os.path.join(frontend_path, "dashboard.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

history: deque = deque(maxlen=100)
connected_clients: list[WebSocket] = []
current_mode: str = "normal"


# -- BUILD PREDICTION PAYLOAD -----------------------------------------------

def build_payload(reading: dict) -> dict:
    # -- REAL MODEL SCORING (0-100 scale) --
    # Single unified model — score once, use for both layers
    score, explanation, top_feature = get_supervised_score(reading)
    cyber_score = score
    physical_score = score
    cyber_top_feature = top_feature
    physical_top_feature = top_feature

    # Audio/Visual layers (not yet implemented)
    audio_score = None
    visual_score = None

    # Fusion expects 0-1 scale inputs
    cyber_01 = cyber_score / 100.0
    physical_01 = physical_score / 100.0

    fusion_score, mode_active = get_risk_score(cyber_01, physical_01, audio_score, visual_score)
    risk_level = classify_risk(cyber_01, physical_01, audio_score, visual_score)

    # SHAP-style explanation
    total = cyber_score + physical_score + 0.01  # avoid div by zero

    cyber_label = cyber_top_feature if cyber_top_feature else (
        "Network command anomaly" if cyber_score > 50 else "Network baseline normal"
    )
    physical_label = physical_top_feature if physical_top_feature else "Sensor deviation"

    shap = ShapExplanation(
        cyber=ShapFeature(
            label=cyber_label,
            pct=round(cyber_score / total * 100, 1),
        ),
        audio=ShapFeature(
            label="Acoustic baseline normal",
            pct=0.0,
        ),
        visual=ShapFeature(
            label="Visual feed normal",
            pct=0.0,
        ),
    )

    top_feature = cyber_label if cyber_score >= physical_score else physical_label
    action = get_recommended_action(risk_level, top_feature)

    payload = PredictionPayload(
        cyber_score=round(cyber_score, 1),       # already 0-100
        physical_score=round(physical_score, 1),  # already 0-100
        audio_score=None,
        visual_score=None,
        fusion_score=fusion_score,
        risk_level=risk_level,
        mode_active=mode_active,
        shap_explanation=shap,
        recommended_action=action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return payload.model_dump()


# -- WEBSOCKET ---------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            reading = get_current_reading()
            payload = build_payload(reading)
            history.append(payload)
            # Broadcast to all clients
            disconnected = []
            for client in connected_clients:
                try:
                    await client.send_json(payload)
                except Exception:
                    disconnected.append(client)
            for client in disconnected:
                if client in connected_clients:
                    connected_clients.remove(client)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)


# -- REST ENDPOINTS ----------------------------------------------------------

@app.post("/simulate/{mode}")
async def simulate(mode: str):
    global current_mode
    valid = ("normal", "cyber_attack", "physical_fault", "critical")
    if mode not in valid:
        return {"error": f"Invalid mode. Choose from: {valid}"}
    current_mode = mode
    if not USE_REAL_PLC:
        stream_set_mode(mode)
    return {"status": "ok", "mode": mode}


@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        mode=current_mode,
        use_real_plc=USE_REAL_PLC,
        layers_active={
            "cyber": True,
            "physical": True,
            "audio": False,
            "visual": False,
        },
    )


@app.get("/history")
async def get_history():
    return list(history)
