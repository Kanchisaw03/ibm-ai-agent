from __future__ import annotations

import re
from typing import List, Optional

from transformers import pipeline

from .config import settings


class NLPProcessor:
    """Lightweight NLP utilities: cleaning, sentiment, grievance extraction."""

    def __init__(self) -> None:
        self._sentiment = pipeline(
            task="sentiment-analysis",
            model=settings.sentiment_model_name,
            tokenizer=settings.sentiment_model_name,
        )

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"http[s]?://\S+", " ", text)
        text = re.sub(r"[@#]\w+", " ", text)
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def analyze_sentiment(self, text: str) -> str:
        result = self._sentiment(text[:512])[0]
        label = result.get("label", "").lower()
        if "neutral" in label:
            return "Neutral"
        if "negative" in label:
            return "Negative"
        if "positive" in label:
            return "Positive"
        return "Neutral"

    @staticmethod
    def _sentences(text: str) -> List[str]:
        # naive splitter to avoid heavy deps
        parts = re.split(r"(?<=[\.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def is_cultural_heritage(text: str) -> bool:
        t = text.lower()
        heritage_terms = [
            "heritage",
            "cultural",
            "culture",
            "historical",
            "history",
            "monument",
            "temple",
            "mosque",
            "church",
            "museum",
            "festival",
            "archaeology",
            "archaeological",
            "fort",
            "palace",
            "site",
            "landmark",
        ]
        return any(term in t for term in heritage_terms)

    def extract_grievance(self, text: str) -> str:
        """Return a short phrase capturing main grievance/request/suggestion.

        Heuristic: find sentence containing any governance/heritage keyword; else first sentence/20 words.
        """
        lowered = text.lower()
        keywords = [
            # governance
            "road",
            "roads",
            "pothole",
            "water",
            "drain",
            "drainage",
            "sewage",
            "electric",
            "electricity",
            "power",
            "garbage",
            "waste",
            "sanitation",
            "health",
            "hospital",
            "school",
            "safety",
            "police",
            "lighting",
            "streetlight",
            "park",
            "tree",
            "corruption",
            "permit",
            "license",
            # heritage
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
        ]

        for sentence in self._sentences(text):
            s_low = sentence.lower()
            if any(k in s_low for k in keywords):
                return sentence[:200]

        # Fallback: first ~20 words
        words = text.split()
        return " ".join(words[:20])


# Singleton-like instance for reuse
nlp = NLPProcessor()


