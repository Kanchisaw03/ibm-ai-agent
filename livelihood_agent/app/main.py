import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .schemas import LivelihoodInput, LivelihoodOutput, HealthStatus, setup_logging
from .llm_agent import create_llm_chain, build_prompt_vars, parse_llm_output
from .ml_predictor import analyze_numeric
from .map_generator import generate_livelihood_map, MAP_DIR
from .config import settings


# Initialize logging similar to housing agent
setup_logging()
logger = logging.getLogger(__name__)


llm_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_chain
    logger.info("Livelihood Agent startup...")
    llm_chain = create_llm_chain()
    yield
    logger.info("Livelihood Agent shutdown...")


app = FastAPI(
    title="Livelihood & Informal Economy Agent",
    description="Analyzes livelihoods and informal economy dynamics.",
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


@app.get("/map/{map_id}", tags=["Maps"])
async def get_map(map_id: str):
    map_path = os.path.join(MAP_DIR, f"{map_id}.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="Map not found")
    return FileResponse(map_path, media_type="text/html")


@app.post(
    "/analyze_livelihood",
    response_model=LivelihoodOutput,
    status_code=status.HTTP_201_CREATED,
    tags=["Livelihood Agent"],
)
async def analyze_livelihood(
    payload: LivelihoodInput,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    try:
        if x_api_key != settings.API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
        # 1) Numeric/statistical analysis
        numeric = analyze_numeric(payload.dict())

        # 2) LLM recommendations
        llm_vars = build_prompt_vars(payload.dict())
        llm_resp = llm_chain.invoke(llm_vars)
        # Some pipelines return dict with 'text' or 'generated_text'; handle both
        if isinstance(llm_resp, dict):
            llm_text = llm_resp.get("text") or llm_resp.get("generated_text") or ""
        else:
            llm_text = str(llm_resp or "")
        parsed = parse_llm_output(llm_text or "")

        # 3) Priority zones
        priority_zones = parsed.get("priority_zones") or []
        if not priority_zones:
            # fallback: construct a zone hint from location
            priority_zones = [str(payload.location)]

        # 4) Predicted risks combine LLM and numeric heuristics
        predicted_risks = list({*parsed.get("predicted_risks", []), *numeric.get("predicted_risks", [])})

        # 5) Fallback recommendation/rationale if LLM response was weak
        recommendation = (parsed.get("recommendation") or "").strip()
        rationale = (parsed.get("rationale") or "").strip()
        def looks_like_echo(text: str) -> bool:
            t = (text or "").lower()
            return any(k in t for k in ["\"location\"", "income_distribution", "key_industries", "skills_profile", "informal_sector_size"]) and len(t) > 20

        if not recommendation or looks_like_echo(recommendation):
            industries = ", ".join(payload.key_industries[:3]) or "informal workers"
            location_hint = str(payload.location)
            recommendation = (
                f"Launch targeted skills programs and microcredit for {industries}; "
                f"formalize vendor zones and improve basic services near {location_hint}."
            )
        if not rationale or looks_like_echo(rationale):
            low = float(payload.income_distribution.get("low", 0))
            informal = float(payload.informal_sector_size)
            rationale = (
                f"Population {payload.population:,} with low-income share {int(low*100)}% and "
                f"informal sector {int(informal*100)}% indicates vulnerability; recommended actions reduce risk and support livelihoods."
            )

        # 6) Map generation (best-effort)
        # unique map id per request
        map_id = f"livelihood_{abs(hash(str(payload.dict())))}"
        map_path: Optional[str] = generate_livelihood_map(map_id, priority_zones)
        map_url: Optional[str] = f"/map/{map_id}" if map_path else None

        result: LivelihoodOutput = LivelihoodOutput(
            recommendation=recommendation,
            rationale=rationale,
            priority_zones=priority_zones,
            predicted_risks=predicted_risks,
            map_url=map_url,
        )
        return result
    except Exception as e:
        logger.error(f"Error in /analyze_livelihood: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not analyze livelihood.")


# --- Minimal Frontend ---
INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Livelihood & Informal Economy Agent</title>
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
  <h2>Livelihood & Informal Economy Agent</h2>
  <form id=\"f\">
    <label>API Key <input name=\"api_key\" placeholder=\"Your API key\" /></label>
    <label>Location <input name=\"location\" placeholder=\"City or lat,lon\" required /></label>
    <div class=\"row\">
      <label>Population <input name=\"population\" type=\"number\" min=\"0\" value=\"0\" required /></label>
      <label>Informal sector size (0-1) <input name=\"informal_sector_size\" type=\"number\" step=\"0.01\" min=\"0\" max=\"1\" value=\"0.2\" required /></label>
    </div>
    <label>Income distribution JSON
      <textarea name=\"income_distribution\" rows=\"2\">{\"low\":0.4,\"mid\":0.4,\"high\":0.2}</textarea>
    </label>
    <label>Key industries (comma-separated)
      <input name=\"key_industries\" placeholder=\"retail, manufacturing, services\" />
    </label>
    <label>Skills profile (comma-separated)
      <input name=\"skills_profile\" placeholder=\"carpentry, tailoring, food services\" />
    </label>
    <label>Policy context
      <textarea name=\"policy_context\" rows=\"3\" placeholder=\"Summarize relevant policies...\"></textarea>
    </label>
    <button>Analyze</button>
  </form>
  <div id=\"out\" class=\"result\"></div>
  <script>
    const f = document.getElementById('f');
    const out = document.getElementById('out');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const data = new FormData(f);
      let income_distribution = {};
      try { income_distribution = JSON.parse(data.get('income_distribution')); } catch {}
      const payload = {
        location: data.get('location'),
        population: Number(data.get('population')),
        income_distribution,
        key_industries: (data.get('key_industries')||'').split(',').map(s=>s.trim()).filter(Boolean),
        informal_sector_size: Number(data.get('informal_sector_size')),
        skills_profile: (data.get('skills_profile')||'').split(',').map(s=>s.trim()).filter(Boolean),
        policy_context: data.get('policy_context')||''
      };
      out.innerHTML = '<span class="muted">Running analysis...</span>';
      const apiKey = data.get('api_key')||'';
      const headers = {'Content-Type':'application/json'};
      if (apiKey) headers['X-API-Key'] = apiKey;
      const res = await fetch('/analyze_livelihood', { method:'POST', headers, body: JSON.stringify(payload)});
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


