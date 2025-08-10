import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .schemas import HousingInput, HousingOutput, HealthStatus, setup_logging
from .llm_agent import create_llm_chain, build_prompt_vars, parse_llm_output
from .ml_predictor import predict_risk_level, load_or_init_model, find_similar_risk_points
from .map_generator import generate_housing_map, MAP_DIR
from .config import settings


# Initialize logging similar to existing agents
setup_logging()
logger = logging.getLogger(__name__)


llm_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_chain
    logger.info("Housing Agent startup...")
    load_or_init_model()
    llm_chain = create_llm_chain()
    yield
    logger.info("Housing Agent shutdown...")


app = FastAPI(
    title="Housing Risk Assessment Agent",
    description="Analyzes housing risk and provides recommendations using local ML + LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus, tags=["Monitoring"])
async def health_check():
    return {"status": "ok"}


@app.get("/map/{filename}", tags=["Maps"])
async def get_map(filename: str):
    map_path = os.path.join(MAP_DIR, f"{filename}.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="Map not found")
    return FileResponse(map_path, media_type="text/html")


@app.post(
    "/analyze_housing",
    response_model=HousingOutput,
    status_code=status.HTTP_201_CREATED,
    tags=["Housing Agent"],
)
async def analyze_housing(
    payload: HousingInput,
    background_tasks: BackgroundTasks,
):
    """Process input, predict risk, call LLM reasoning, and generate a map."""
    try:
        # 1) Predict risk level via local model
        risk = predict_risk_level(payload.dict())
        risk_level = risk.get("risk_level", "medium")
        predicted_risks = risk.get("predicted_risks", [])

        # 2) LLM recommendation and rationale
        llm_vars = build_prompt_vars(payload.dict())
        llm_resp = llm_chain.invoke(llm_vars)
        if isinstance(llm_resp, dict):
            llm_text = llm_resp.get("text") or llm_resp.get("generated_text") or ""
        else:
            llm_text = str(llm_resp or "")
        parsed = parse_llm_output(llm_text)
        recommendation = (parsed.get("recommendation") or "").strip()
        rationale = (parsed.get("rationale") or "").strip()

        # 3) Select priority zones from dataset by same cluster risk
        lat0, lon0 = float(payload.latitude), float(payload.longitude)
        cluster_points = find_similar_risk_points(payload.dict(), max_points=3)

        # 4) Map generation
        map_id = f"housing_{abs(hash(str(payload.dict())))}"
        map_path: Optional[str] = generate_housing_map(map_id, (lat0, lon0), risk_level, cluster_points)
        map_url: Optional[str] = f"/map/{map_id}" if map_path else None

        result: HousingOutput = HousingOutput(
            recommendation=recommendation,
            rationale=rationale,
            priority_zones=[f"{lat:.5f},{lon:.5f}" for lat, lon in cluster_points],
            predicted_risks=predicted_risks,
            map_url=map_url,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /analyze_housing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not analyze housing.")


# --- Minimal Frontend ---
INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Housing Risk Assessment Agent</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    form { display: grid; gap: 8px; max-width: 720px; }
    textarea, input { width: 100%; padding: 8px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .result { margin-top: 16px; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }
    iframe { width: 100%; height: 460px; border: 1px solid #ccc; border-radius: 8px; }
    .muted { color: #666; }
  </style>
  </head>
<body>
  <h2>Housing Risk Assessment Agent</h2>
  <form id=\"f\">
    <label>Location Name <input name=\"location_name\" placeholder=\"Neighborhood or area\" required /></label>
    <div class=\"row\">
      <label>Latitude <input name=\"latitude\" type=\"number\" step=\"0.00001\" required /></label>
      <label>Longitude <input name=\"longitude\" type=\"number\" step=\"0.00001\" required /></label>
    </div>
    <div class=\"row\">
      <label>Housing condition (0-100) <input name=\"housing_condition_score\" type=\"number\" min=\"0\" max=\"100\" value=\"50\" required /></label>
      <label>Population density (people/km²) <input name=\"population_density\" type=\"number\" min=\"0\" value=\"1000\" required /></label>
    </div>
    <div class=\"row\">
      <label>Low-income households (%) <input name=\"low_income_household_percentage\" type=\"number\" min=\"0\" max=\"100\" value=\"30\" required /></label>
      <label>Access to services (0-100) <input name=\"access_to_services_score\" type=\"number\" min=\"0\" max=\"100\" value=\"60\" required /></label>
    </div>
    <button>Analyze</button>
  </form>
  <div id=\"out\" class=\"result\"></div>
  <script>
    const f = document.getElementById('f');
    const out = document.getElementById('out');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const data = new FormData(f);
      const payload = {
        location_name: data.get('location_name'),
        latitude: Number(data.get('latitude')),
        longitude: Number(data.get('longitude')),
        housing_condition_score: Number(data.get('housing_condition_score')),
        population_density: Number(data.get('population_density')),
        low_income_household_percentage: Number(data.get('low_income_household_percentage')),
        access_to_services_score: Number(data.get('access_to_services_score')),
      };
      out.innerHTML = '<span class="muted">Running analysis...</span>';
      const headers = {'Content-Type':'application/json'};
      const res = await fetch('/analyze_housing', { method:'POST', headers, body: JSON.stringify(payload)});
      const json = await res.json();
      out.innerHTML = `
        <div><strong>Recommendation</strong>: ${json.recommendation||''}</div>
        <div><strong>Rationale</strong>: ${json.rationale||''}</div>
        <div><strong>Priority Zones</strong>: ${(json.priority_zones||[]).join(', ')}</div>
        <div><strong>Predicted Risks</strong>: ${(json.predicted_risks||[]).join(', ')}</div>
        ${json.map_url ? `<div style=\"margin-top:12px\"><iframe src=\"${json.map_url}\"></iframe></div>` : ''}
      `;
    });
  </script>
 </body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def index():
    return HTMLResponse(INDEX_HTML)


