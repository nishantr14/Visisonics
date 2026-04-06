"""OTShield FastAPI backend — WebSocket broadcast + REST endpoints."""

import asyncio
import random
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from fusion import get_risk_score, classify_risk, get_recommended_action
from api_models import PredictionPayload, StatusResponse, ShapExplanation, ShapFeature

# ── DATA SOURCE SWITCH ──────────────────────────────
USE_REAL_PLC = False

if USE_REAL_PLC:
    from data_collector import get_current_reading
else:
    from fake_plc_stream import get_current_reading, set_mode as stream_set_mode

# ────────────────────────────────────────────────────

app = FastAPI(title="OTShield API", version="0.1.0")

# ── STATIC FILE SERVING ─────────────────────────────
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


# ── PLACEHOLDER SCORING FUNCTIONS ───────────────────

def _placeholder_cyber(reading: dict) -> float:
    """Placeholder cyber score: returns 0-1 based on ATT_FLAG + noise."""
    base = 0.7 if reading.get('ATT_FLAG', 0) == 1 else 0.15
    return round(min(1.0, max(0.0, base + random.gauss(0, 0.08))), 4)


def _placeholder_physical(reading: dict) -> float:
    """Placeholder physical score: derived from sensor deviations."""
    l_avg = (reading.get('L_T1', 5) + reading.get('L_T2', 5) + reading.get('L_T3', 5)) / 3
    deviation = abs(l_avg - 5.0) / 5.0
    base = min(1.0, deviation + 0.1)
    if reading.get('ATT_FLAG', 0) == 1:
        base = max(base, 0.5)
    return round(min(1.0, max(0.0, base + random.gauss(0, 0.05))), 4)


# ── BUILD PREDICTION PAYLOAD ────────────────────────

def build_payload(reading: dict) -> dict:
    # ── CYBER LAYER HOOK ──────────────────────────────
    # Replace this block with: from cyber_layer import get_cyber_score
    cyber_score = _placeholder_cyber(reading)

    # ── PHYSICAL LAYER HOOK ───────────────────────────
    # Replace this block with: from physical_layer import get_physical_score
    physical_score = _placeholder_physical(reading)

    # ── AUDIO LAYER HOOK ──────────────────────────────
    # Replace this block with: from acoustic_layer import get_acoustic_score
    audio_score = None

    # ── VISUAL LAYER HOOK ─────────────────────────────
    # Replace this block with: from visual_layer import get_visual_score
    visual_score = None

    fusion_score, mode_active = get_risk_score(cyber_score, physical_score, audio_score, visual_score)
    risk_level = classify_risk(cyber_score, physical_score, audio_score, visual_score)

    # SHAP-style explanation (placeholder percentages)
    total = cyber_score + physical_score + (audio_score or 0) + (visual_score or 0)
    if total == 0:
        total = 1.0
    shap = ShapExplanation(
        cyber=ShapFeature(
            label="Network command anomaly" if cyber_score > 0.5 else "Network baseline normal",
            pct=round(cyber_score / total * 100, 1),
        ),
        audio=ShapFeature(
            label="Acoustic deviation detected" if (audio_score or 0) > 0.5 else "Acoustic baseline normal",
            pct=round((audio_score or 0) / total * 100, 1),
        ),
        visual=ShapFeature(
            label="Visual anomaly flagged" if (visual_score or 0) > 0.5 else "Visual feed normal",
            pct=round((visual_score or 0) / total * 100, 1),
        ),
    )

    top_feature = shap.cyber.label if cyber_score >= physical_score else "Sensor deviation"
    action = get_recommended_action(risk_level, top_feature)

    payload = PredictionPayload(
        cyber_score=round(cyber_score * 100, 1),
        physical_score=round(physical_score * 100, 1),
        audio_score=round(audio_score * 100, 1) if audio_score is not None else None,
        visual_score=round(visual_score * 100, 1) if visual_score is not None else None,
        fusion_score=fusion_score,
        risk_level=risk_level,
        mode_active=mode_active,
        shap_explanation=shap,
        recommended_action=action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return payload.model_dump()


# ── WEBSOCKET ───────────────────────────────────────

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


# ── REST ENDPOINTS ──────────────────────────────────

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
