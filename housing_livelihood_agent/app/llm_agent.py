import logging
import re
from typing import Dict
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)

def create_llm_chain() -> LLMChain:
    model_id = "google/flan-t5-small"  # small, free, local model
    
    logger.info(f"Loading local model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3
    )
    
    prompt_template = """
You are an expert urban planning assistant with access to mapping tools.

Your job:
1. Recommend a relocation zone that can be plotted on a map (use either a clear area name or explicit latitude,longitude).
2. Give a concise rationale in **no more than two sentences** that references flood risk and/or the vendor's income level.

⚠️ Output **must** follow this exact two-line format (do not add anything else):

Recommendation: <Zone name OR lat,lon>
Rationale: <One or two sentences>

Vendor data:
- Income Level: {income_level}
- Flood Risk: {flood_risk_level}
- Current Location: Lat {current_lat}, Lon {current_lon}
"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    logger.info(f"LLMChain created successfully with local model: {model_id}")
    return LLMChain(llm=llm, prompt=prompt)


def parse_llm_output(output: str) -> Dict[str, str]:
    try:
        # Try strict regex match
        rec_match = re.search(r"Recommendation[:\-]\s*(.*)", output, re.IGNORECASE)
        rat_match = re.search(r"Rationale[:\-]\s*(.*)", output, re.IGNORECASE)

        recommendation = rec_match.group(1).strip() if rec_match else "Not specified"
        rationale = rat_match.group(1).strip() if rat_match else "No clear rationale provided."

        # Fallback guess if still not found
        if recommendation == "Not specified" and output:
            first_line = output.strip().split("\n")[0]
            if first_line:
                recommendation = first_line.strip()
        
        return {"recommended_zone": recommendation, "rationale": rationale}
    
    except Exception:
        logger.warning(f"Could not parse LLM output cleanly: {output}")
        return {"recommended_zone": "Not specified", "rationale": output.strip()}
