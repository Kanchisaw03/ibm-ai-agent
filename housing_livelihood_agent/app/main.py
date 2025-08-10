import logging
from contextlib import asynccontextmanager
import datetime

from fastapi import FastAPI, Depends, HTTPException, status, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .config import settings
from .hf_db import add_plan_to_hf_dataset
from .ml_predictor import load_model, score_plan
from .llm_agent import create_llm_chain, parse_llm_output
from .map_generator import generate_map, MAP_DIR
from .schemas import (
    RelocationInput,
    RelocationPlanOutput,
    HealthStatus,
    setup_logging,
)

# Configure structured logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Global LLM Chain ---
llm_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global llm_chain
    logger.info("Application startup...")
    load_model()
    llm_chain = create_llm_chain()
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")

# Create and configure the FastAPI application
app = FastAPI(
    title="Housing & Livelihood AI Agent (Hugging Face Edition)",
    description="An agent using Hugging Face LLMs for urban planning.",
    version="4.2.0", # Version bump for sync workaround
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)

# Serve static map files
from fastapi.responses import FileResponse
import os

@app.get("/map/{vendor_id}", tags=["Maps"])
async def get_map(vendor_id: str):
    map_path = os.path.join(MAP_DIR, f"{vendor_id}.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="Map not found")
    return FileResponse(map_path, media_type="text/html")

# --- Security Dependency ---
async def verify_api_key(x_api_key: str = Header(..., description="Your secret API key")):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# --- API Endpoints ---
@app.get("/health", response_model=HealthStatus, tags=["Monitoring"])
async def health_check():
    return {"status": "ok"}

@app.post(
    "/relocate",
    response_model=RelocationPlanOutput,
    status_code=status.HTTP_201_CREATED,
    tags=["Relocation Agent"],
    dependencies=[Depends(verify_api_key)],
)
async def relocate_vendor(
    relocation_input: RelocationInput,
    background_tasks: BackgroundTasks,
):
    """
    Generates a relocation plan using a Hugging Face LLM and scores it with a local ML model.
    """
    logger.info(f"Relocation request received for vendor: {relocation_input.vendor_id}")

    try:
        # 1. Get recommendation from the LLM chain (using synchronous 'invoke')
        # FastAPI will run this in a thread pool to avoid blocking the event loop.
        llm_output = llm_chain.invoke(relocation_input.dict())
        llm_output_str = llm_output['text']  # Extract the 'text' key
        parsed_response = parse_llm_output(llm_output_str)

        # 2. Score the plan using the ML model
        ml_score = score_plan(relocation_input.dict())

        # 3. Assemble the final plan
        # 3a. Attempt to generate a map (only if coordinates present)
        map_path = generate_map(relocation_input.vendor_id, parsed_response["recommended_zone"])
        map_url = None
        if map_path:
            map_url = f"/map/{relocation_input.vendor_id}"

        # 3b. Assemble the final plan
        final_plan = RelocationPlanOutput(
            vendor_id=relocation_input.vendor_id,
            recommended_zone=parsed_response["recommended_zone"],
            rationale=parsed_response["rationale"],
            ml_score=ml_score,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            map_url=map_url,
        )

        # 4. Save to DB in the background
        background_tasks.add_task(add_plan_to_hf_dataset, final_plan.dict())
        
        logger.info(f"Successfully generated relocation plan for vendor: {relocation_input.vendor_id}")
        return final_plan

    except Exception as e:
        logger.error(f"Error during relocation process for vendor {relocation_input.vendor_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not process the relocation request.")
