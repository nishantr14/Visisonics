"""Pydantic models for OTShield API."""

from pydantic import BaseModel


class PredictionPayload(BaseModel):
    cyber_score: float
    physical_score: float
    fusion_score: float
    cyber_risk_label: str
    physical_risk_label: str
    final_risk_label: str
    explanation: str
    timestamp: str


class StatusResponse(BaseModel):
    mode: str
    use_real_plc: bool
    layers_active: dict
