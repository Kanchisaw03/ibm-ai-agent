import json
import logging
from typing import Dict

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = (
    """
You are an expert urban housing analyst.  
Based on the following housing data, provide:  
1. A one-sentence **relocation or upgrade recommendation**.  
2. A one-sentence **rationale** explaining the risk factors.  

**Housing Data:**  
- Condition Score: {housing_condition_score}  
- Population Density: {population_density} people/km²  
- Low-Income Households: {low_income_household_percentage}%  
- Access to Services Score: {access_to_services_score}  
- Location: Lat {latitude}, Lon {longitude} ({location_name})  

**Output Format:**  
Recommendation: [Your recommendation]  
Rationale: [Your reasoning]  
"""
)


def create_llm_chain() -> LLMChain:
    """Create a local FLAN-T5 LLM chain for housing analysis."""

    model_id = "google/flan-t5-small"
    logger.info(f"Loading local model for housing agent: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=192,
        temperature=0.3,
        do_sample=True,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = HuggingFacePipeline(pipeline=pipe)
    return LLMChain(llm=llm, prompt=prompt)


def build_prompt_vars(payload: Dict) -> Dict[str, str]:
    return {
        "location_name": str(payload.get("location_name", "")),
        "latitude": str(payload.get("latitude", "")),
        "longitude": str(payload.get("longitude", "")),
        "housing_condition_score": str(payload.get("housing_condition_score", "")),
        "population_density": str(payload.get("population_density", "")),
        "low_income_household_percentage": str(payload.get("low_income_household_percentage", "")),
        "access_to_services_score": str(payload.get("access_to_services_score", "")),
    }


def parse_llm_output(output_text: str) -> Dict:
    """Parse two labeled lines into fields with safe fallbacks."""
    text = (output_text or "").strip()
    rec = ""
    rat = ""
    for line in text.splitlines():
        lower = line.lower().strip()
        if lower.startswith("recommendation:") and not rec:
            rec = line.split(":", 1)[1].strip()
        elif lower.startswith("rationale:") and not rat:
            rat = line.split(":", 1)[1].strip()

    if not rec:
        # Try JSON fallback if model emitted JSON
        try:
            obj = json.loads(text)
            rec = str(obj.get("recommendation", "")).strip()
            rat = str(obj.get("rationale", "")).strip() or rat
        except Exception:
            pass

    return {"recommendation": rec, "rationale": rat}


