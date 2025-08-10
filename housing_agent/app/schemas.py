import logging
import sys
from typing import Optional, List

from pydantic import BaseModel, Field


class HousingInput(BaseModel):
    """Input schema for /analyze_housing.

    location_name: human-readable name for the area.
    latitude/longitude: coordinates of the location.
    Scores are expected on 0–100 scale as specified.
    """

    location_name: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    housing_condition_score: float = Field(..., ge=0, le=100)
    population_density: float = Field(..., ge=0)
    low_income_household_percentage: float = Field(..., ge=0, le=100)
    access_to_services_score: float = Field(..., ge=0, le=100)


class HousingOutput(BaseModel):
    recommendation: str
    rationale: str
    priority_zones: List[str]
    predicted_risks: List[str]
    map_url: Optional[str]


class HealthStatus(BaseModel):
    status: str


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON-like string for easy ingestion."""

    def format(self, record):  # type: ignore[override]
        return (
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"message": "%(message)s", "name": "%(name)s"}'
        ) % {
            "asctime": self.formatTime(record, self.datefmt),
            "levelname": record.levelname,
            "message": record.getMessage().replace('"', "'"),
            "name": record.name,
        }


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


