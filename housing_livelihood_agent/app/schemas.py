import logging
import sys
from datetime import datetime
from typing import Literal, Dict, Any, Optional

from pydantic import BaseModel, Field

# --- Custom Types ---
IncomeLevel = Literal["low", "mid", "high"]
FloodRiskLevel = Literal["low", "moderate", "high"]

# --- Input Schemas ---
class RelocationInput(BaseModel):
    """
    Schema for the input to the /relocate endpoint.
    """
    vendor_id: str
    income_level: IncomeLevel
    current_lat: float = Field(..., ge=-90.0, le=90.0)
    current_lon: float = Field(..., ge=-180.0, le=180.0)
    flood_risk_level: FloodRiskLevel

# --- Output Schemas ---
class RelocationPlanOutput(BaseModel):
    """
    Final relocation plan including map URL.
    """
    vendor_id: str
    recommended_zone: str
    rationale: str
    ml_score: float
    timestamp: datetime
    map_url: Optional[str] = None

class HealthStatus(BaseModel):
    """Schema for health and readiness check responses."""
    status: str

# --- Logging Setup ---
class JsonFormatter(logging.Formatter):
    """Formats log records as JSON."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return str(log_record)

def setup_logging():
    """Configures root logger for structured JSON output."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "name": "%(name)s"}'
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
