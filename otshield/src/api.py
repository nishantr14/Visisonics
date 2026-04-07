"""OTShield FastAPI backend -- dual-layer scoring + WebSocket broadcast."""

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

# Independent scorers — unified deep learning architecture
from physical_layer import get_physical_score
from cyber_layer import get_cyber_score

# -- DATA SOURCE SWITCH -----------------------------------------------------
USE_REAL_PLC = False

if USE_REAL_PLC:
    from data_collector import get_current_reading
    # No cyber stream from real PLC — would need a network tap
    get_physical_reading = get_current_reading
    get_cyber_reading = None
else:
    from fake_plc_stream import (
        get_physical_reading,
        get_cyber_reading,
        set_mode as stream_set_mode,
    )

# ---------------------------------------------------------------------------

app = FastAPI(title="OTShield API", version="0.3.0")

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

def build_payload() -> dict:
    """Score both layers independently and fuse results."""

    # ── PHYSICAL LAYER (BATADAL Isolation Forest) ──
    physical_reading = get_physical_reading()
    physical_score_01, physical_expl, physical_top = get_physical_score(physical_reading)
    physical_score = physical_score_01 * 100.0

    # ── CYBER LAYER (TON/UNSW Deep Learning) ──
    if get_cyber_reading is not None:
        network_flow = get_cyber_reading()
        cyber_score_01, cyber_expl, cyber_top = get_cyber_score(network_flow)
        cyber_score = cyber_score_01 * 100.0
    else:
        cyber_score, cyber_score_01, cyber_expl, cyber_top = 0.0, 0.0, "No network feed", "N/A"

    # Audio/Visual layers (not yet implemented)
    audio_score = None
    visual_score = None

    # ── FUSION (expects 0-1 scale) ──
    fusion_score, mode_active = get_risk_score(cyber_score_01, physical_score_01, audio_score, visual_score)
    risk_level = classify_risk(cyber_score_01, physical_score_01, audio_score, visual_score)

    # ── SHAP-style explanation ──
    total = cyber_score + physical_score + 0.01  # avoid div-by-zero
    cyber_pct = round(cyber_score / total * 100, 1)
    physical_pct = round(physical_score / total * 100, 1)

    shap = ShapExplanation(
        cyber=ShapFeature(
            label=f"{cyber_top} -- {cyber_expl}" if cyber_score > 5 else "Network baseline normal",
            pct=cyber_pct,
        ),
        physical=ShapFeature(
            label=f"{physical_top} -- {physical_expl}" if physical_score > 5 else "Sensor telemetry nominal",
            pct=physical_pct,
        ),
        audio=ShapFeature(label="Acoustic baseline normal", pct=0.0),
        visual=ShapFeature(label="Visual feed normal", pct=0.0),
    )

    top_feature = cyber_top if cyber_score >= physical_score else physical_top
    action = get_recommended_action(risk_level, top_feature)

    payload = PredictionPayload(
        cyber_score=round(cyber_score, 1),
        physical_score=round(physical_score, 1),
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
            payload = build_payload()
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
