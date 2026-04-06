"""Pydantic models for OTShield API."""

from pydantic import BaseModel
from typing import Optional


class ShapFeature(BaseModel):
    label: str
    pct: float


class ShapExplanation(BaseModel):
    cyber: ShapFeature
    physical: ShapFeature
    audio: ShapFeature
    visual: ShapFeature


class PredictionPayload(BaseModel):
    cyber_score: Optional[float] = None
    physical_score: Optional[float] = None
    audio_score: Optional[float] = None
    visual_score: Optional[float] = None
    fusion_score: float
    risk_level: str
    mode_active: str
    shap_explanation: ShapExplanation
    recommended_action: str
    timestamp: str


class StatusResponse(BaseModel):
    mode: str
    use_real_plc: bool
    layers_active: dict
