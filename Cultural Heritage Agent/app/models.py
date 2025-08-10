from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    citizen_id: str
    feedback_text: str
    location_lat: float
    location_lon: float
    timestamp: datetime


class AnalysisResult(BaseModel):
    sentiment: str
    grievance_category: str
    recommendation: str
    map_url: str


class FeedbackResponse(BaseModel):
    citizen_id: str
    result: AnalysisResult


class BulkLoadResponse(BaseModel):
    count: int
    map_url: str


