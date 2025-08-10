import logging
import sys
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LivelihoodInput(BaseModel):
    """Input schema for /analyze_livelihood.

    location: can be a city string or a "lat,lon" pair
    """

    location: str = Field(..., description="City name or 'lat,lon' pair")
    population: int = Field(..., ge=0)
    income_distribution: Dict[str, float] = Field(
        ..., description="Distribution of income brackets (e.g., {'low': 0.4, 'mid': 0.4, 'high': 0.2})"
    )
    key_industries: List[str]
    informal_sector_size: float = Field(..., ge=0)
    skills_profile: List[str]
    policy_context: str


class LivelihoodOutput(BaseModel):
    recommendation: str
    rationale: str
    priority_zones: List[str]
    predicted_risks: List[str]
    map_url: Optional[str]


class HealthStatus(BaseModel):
    status: str


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON-like string for easy ingestion."""

    def format(self, record):
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


