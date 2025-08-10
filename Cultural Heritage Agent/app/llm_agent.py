from __future__ import annotations

from typing import Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from .config import settings


class LLMAgent:
    """Thin wrapper around a local text2text model for structured guidance output."""

    def __init__(self) -> None:
        # Public models don't require tokens, but pass if provided
        self.generator = pipeline(
            task="text2text-generation",
            model=settings.hf_model_name,
            tokenizer=settings.hf_model_name,
            model_kwargs={"torch_dtype": "auto"},
        )

    @staticmethod
    def build_prompt(
        feedback_text: str,
        heuristic_sentiment: str,
        extracted_grievance: str,
        is_cultural_heritage: bool,
    ) -> str:
        categories = (
            "Infrastructure, Public Services, Sanitation, Environment, Safety, Governance/Corruption, Cultural Heritage, Other"
        )
        instruction = (
            "You are a civic governance assistant. Analyze the citizen feedback and respond in EXACTLY three lines with this format:\n"
            "Sentiment: [Positive/Negative/Neutral]\n"
            "Grievance Category: [one of: "
            + categories
            + "]\n"
            "Recommendation: [one concise, actionable step for local authorities]"
        )

        context = (
            f"Heuristic Sentiment: {heuristic_sentiment}. "
            f"Extracted Grievance: {extracted_grievance}. "
            f"Cultural Heritage Flag: {is_cultural_heritage}. "
            f"Feedback: {feedback_text}"
        )
        return instruction + "\n\n" + context

    @staticmethod
    def _parse_response(text: str) -> Dict[str, str]:
        # Robust parse of the three lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        result: Dict[str, str] = {"sentiment": "", "category": "", "recommendation": ""}
        for line in lines:
            lower = line.lower()
            if lower.startswith("sentiment:"):
                result["sentiment"] = line.split(":", 1)[1].strip().capitalize()
            elif lower.startswith("grievance category:"):
                result["category"] = line.split(":", 1)[1].strip()
            elif lower.startswith("recommendation:"):
                result["recommendation"] = line.split(":", 1)[1].strip()
        return result

    def generate_structured(self, feedback_text: str, heuristic_sentiment: str, extracted_grievance: str, is_cultural_heritage: bool) -> Dict[str, str]:
        prompt = self.build_prompt(
            feedback_text=feedback_text,
            heuristic_sentiment=heuristic_sentiment,
            extracted_grievance=extracted_grievance,
            is_cultural_heritage=is_cultural_heritage,
        )
        out = self.generator(prompt, max_new_tokens=128)
        text = out[0]["generated_text"] if isinstance(out, list) and out else ""
        return self._parse_response(text)


llm_agent = LLMAgent()


