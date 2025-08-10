from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .llm_agent import llm_agent
from .map_gui import generate_map
from .models import AnalysisResult, BulkLoadResponse, FeedbackRequest, FeedbackResponse
from .nlp_processor import nlp
from . import governance_logic as gov


def configure_logging() -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.log_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# CORS permissive for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static content (maps)
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    # Warm-up small calls (optional)
    try:
        _ = nlp.analyze_sentiment("Warmup text for sentiment.")
    except Exception as exc:
        logger.warning(f"Sentiment warmup failed: {exc}")
    try:
        _ = llm_agent.generate_structured(
            feedback_text="Warmup feedback.",
            heuristic_sentiment="Neutral",
            extracted_grievance="Warmup grievance",
            is_cultural_heritage=False,
        )
    except Exception as exc:
        logger.warning(f"LLM warmup failed: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/analyze_feedback", response_model=FeedbackResponse)
def analyze_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    cleaned = nlp.clean_text(payload.feedback_text)
    heuristic_sentiment = nlp.analyze_sentiment(cleaned)
    extracted = nlp.extract_grievance(cleaned)
    is_heritage = nlp.is_cultural_heritage(cleaned)

    # Rule-based category first
    heuristic_category = gov.categorize_grievance(cleaned)

    # LLM structured suggestion
    llm_out = llm_agent.generate_structured(
        feedback_text=cleaned,
        heuristic_sentiment=heuristic_sentiment,
        extracted_grievance=extracted,
        is_cultural_heritage=is_heritage,
    )

    llm_sentiment = llm_out.get("sentiment")
    llm_category = llm_out.get("category")
    llm_reco = llm_out.get("recommendation")

    sentiment, category, recommendation = gov.merge_llm_and_rules(
        llm_sentiment=llm_sentiment,
        llm_category=llm_category,
        llm_recommendation=llm_reco,
        heuristic_category=heuristic_category,
        heuristic_sentiment=heuristic_sentiment,
        is_heritage=is_heritage,
    )

    # Build map for this single feedback
    marker = {
        "lat": payload.location_lat,
        "lon": payload.location_lon,
        "sentiment": sentiment,
        "info": {
            "citizen_id": payload.citizen_id,
            "timestamp": payload.timestamp.isoformat(),
            "sentiment": sentiment,
            "category": category,
            "grievance": extracted,
            "recommendation": recommendation,
        },
    }
    map_url = generate_map([marker])

    result = AnalysisResult(
        sentiment=sentiment, grievance_category=category, recommendation=recommendation, map_url=map_url
    )

    response = FeedbackResponse(citizen_id=payload.citizen_id, result=result)

    # Log API call
    logger.info(
        json.dumps(
            {
                "event": "analyze_feedback",
                "timestamp": datetime.utcnow().isoformat(),
                "citizen_id": payload.citizen_id,
                "sentiment": sentiment,
                "category": category,
                "map_url": map_url,
            }
        )
    )

    return response


@app.get("/load_sample_data", response_model=BulkLoadResponse)
def load_sample_data() -> BulkLoadResponse:
    sample_path = settings.data_dir / "sample_feedback.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample data not found at {sample_path}")

    with sample_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    markers: List[dict] = []
    for item in items:
        try:
            payload = FeedbackRequest(**item)
        except Exception:
            continue

        cleaned = nlp.clean_text(payload.feedback_text)
        heuristic_sentiment = nlp.analyze_sentiment(cleaned)
        extracted = nlp.extract_grievance(cleaned)
        is_heritage = nlp.is_cultural_heritage(cleaned)
        heuristic_category = gov.categorize_grievance(cleaned)

        llm_out = llm_agent.generate_structured(
            feedback_text=cleaned,
            heuristic_sentiment=heuristic_sentiment,
            extracted_grievance=extracted,
            is_cultural_heritage=is_heritage,
        )

        sentiment, category, recommendation = gov.merge_llm_and_rules(
            llm_sentiment=llm_out.get("sentiment"),
            llm_category=llm_out.get("category"),
            llm_recommendation=llm_out.get("recommendation"),
            heuristic_category=heuristic_category,
            heuristic_sentiment=heuristic_sentiment,
            is_heritage=is_heritage,
        )

        markers.append(
            {
                "lat": payload.location_lat,
                "lon": payload.location_lon,
                "sentiment": sentiment,
                "info": {
                    "citizen_id": payload.citizen_id,
                    "timestamp": payload.timestamp.isoformat(),
                    "sentiment": sentiment,
                    "category": category,
                    "grievance": extracted,
                    "recommendation": recommendation,
                },
            }
        )

    map_url = generate_map(markers)

    logger.info(
        json.dumps(
            {
                "event": "load_sample_data",
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(markers),
                "map_url": map_url,
            }
        )
    )

    return BulkLoadResponse(count=len(markers), map_url=map_url)


