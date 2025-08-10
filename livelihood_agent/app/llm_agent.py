import json
import logging
from typing import Dict

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are an urban economic policy advisor. Analyze the following livelihood data and:
- Recommend interventions.
- Identify priority areas.
- Predict future livelihood risks.

ALWAYS respond with STRICT VALID JSON using these exact keys (escape braces are intentional):
{{
  "recommendation": "string",
  "rationale": "string",
  "priority_zones": ["string"],
  "predicted_risks": ["string"]
}}

Return ONLY the JSON object. No explanations.

Data: {json_data}
"""


def create_llm_chain() -> LLMChain:
    model_id = "google/flan-t5-small"
    logger.info(f"Loading local model for livelihood agent: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = HuggingFacePipeline(pipeline=pipe)
    return LLMChain(llm=llm, prompt=prompt)


def build_prompt_vars(payload: Dict) -> Dict[str, str]:
    # Ensure deterministic ordering for readability
    json_str = json.dumps(payload, ensure_ascii=False)
    return {"json_data": json_str}


def parse_llm_output(output_text: str) -> Dict:
    """Parse JSON-first, with resilient fallbacks to label-based or free text."""
    text = (output_text or "").strip()
    # 1) Try strict JSON
    try:
        obj = json.loads(text)
        recommendation = str(obj.get("recommendation", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        priority_zones = [str(x).strip() for x in obj.get("priority_zones", []) if str(x).strip()]
        predicted_risks = [str(x).strip() for x in obj.get("predicted_risks", []) if str(x).strip()]
        return {
            "recommendation": recommendation or "",
            "rationale": rationale or "",
            "priority_zones": priority_zones,
            "predicted_risks": predicted_risks,
        }
    except Exception:
        pass

    # 2) Try to locate a JSON block within text
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            recommendation = str(obj.get("recommendation", "")).strip()
            rationale = str(obj.get("rationale", "")).strip()
            priority_zones = [str(x).strip() for x in obj.get("priority_zones", []) if str(x).strip()]
            predicted_risks = [str(x).strip() for x in obj.get("predicted_risks", []) if str(x).strip()]
            return {
                "recommendation": recommendation or "",
                "rationale": rationale or "",
                "priority_zones": priority_zones,
                "predicted_risks": predicted_risks,
            }
    except Exception:
        pass

    # 3) Fallback: labeled lines
    def _extract(prefix: str) -> str:
        for line in text.splitlines():
            if line.lower().startswith(prefix.lower()):
                return line.split(":", 1)[1].strip()
        return ""
    recommendation = _extract("Recommendation")
    rationale = _extract("Rationale")
    zones_raw = _extract("Priority Zones")
    risks_raw = _extract("Predicted Risks")
    priority_zones = [z.strip() for z in zones_raw.split(",") if z.strip()] if zones_raw else []
    predicted_risks = [r.strip() for r in risks_raw.split(",") if r.strip()] if risks_raw else []
    if recommendation or rationale or priority_zones or predicted_risks:
        return {
            "recommendation": recommendation,
            "rationale": rationale,
            "priority_zones": priority_zones,
            "predicted_risks": predicted_risks,
        }

    # 4) Last resort: use first sentence as recommendation
    first_line = text.splitlines()[0] if text else ""
    return {
        "recommendation": first_line or "",
        "rationale": "",
        "priority_zones": [],
        "predicted_risks": [],
    }


