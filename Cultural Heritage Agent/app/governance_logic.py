from __future__ import annotations

from typing import Tuple


CATEGORY_KEYWORDS = {
    "Infrastructure": [
        "road",
        "pothole",
        "bridge",
        "streetlight",
        "lighting",
        "electric",
        "power",
        "water",
        "drain",
        "sewage",
        "drainage",
        "transport",
        "bus",
        "traffic",
    ],
    "Public Services": [
        "hospital",
        "clinic",
        "health",
        "school",
        "education",
        "permit",
        "license",
        "ration",
        "subsidy",
        "documentation",
    ],
    "Sanitation": ["garbage", "waste", "sanitation", "trash", "cleaning", "sweep"],
    "Environment": ["park", "tree", "lake", "river", "pollution", "air", "noise"],
    "Safety": ["police", "crime", "theft", "safety", "harassment"],
    "Governance/Corruption": ["bribe", "corruption", "nepotism", "delay", "red tape"],
    "Cultural Heritage": [
        "heritage",
        "cultural",
        "monument",
        "temple",
        "mosque",
        "church",
        "museum",
        "festival",
        "historic",
        "archaeolog",
        "fort",
        "palace",
        "landmark",
    ],
}


def categorize_grievance(text: str) -> str:
    lowered = text.lower()
    best_category = "Other"
    best_hits = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in lowered)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category


def default_recommendation(category: str, sentiment: str, is_heritage: bool) -> str:
    if category == "Infrastructure":
        return "Schedule inspection and repair; log a work order with the municipal engineering department."
    if category == "Public Services":
        return "Notify the relevant department; establish a service desk ticket and communicate expected resolution time."
    if category == "Sanitation":
        return "Dispatch sanitation crew; increase waste collection frequency and place public awareness signage."
    if category == "Environment":
        return "Assess environmental impact; initiate cleanup/plantation drive and monitor air/water quality."
    if category == "Safety":
        return "Coordinate with local police; improve lighting and patrol frequency in the area."
    if category == "Governance/Corruption":
        return "Escalate to vigilance cell; ensure transparent e-governance workflows and timelines."
    if category == "Cultural Heritage" or is_heritage:
        return (
            "Engage heritage authorities; schedule preservation assessment and run community awareness/outreach."
        )
    return "Acknowledge receipt; route to appropriate department for triage and follow-up."


def merge_llm_and_rules(
    llm_sentiment: str | None,
    llm_category: str | None,
    llm_recommendation: str | None,
    heuristic_category: str,
    heuristic_sentiment: str,
    is_heritage: bool,
) -> Tuple[str, str, str]:
    # Sentiment: prefer LLM if valid
    sentiment = llm_sentiment if llm_sentiment in {"Positive", "Negative", "Neutral"} else heuristic_sentiment

    # Category: prefer LLM if among known categories
    known_categories = set(CATEGORY_KEYWORDS.keys()) | {"Other"}
    category = llm_category if llm_category in known_categories else heuristic_category

    # Recommendation: prefer LLM if present; else rule-based
    recommendation = (
        llm_recommendation.strip() if llm_recommendation and llm_recommendation.strip() else default_recommendation(category, sentiment, is_heritage)
    )

    return sentiment, category, recommendation


