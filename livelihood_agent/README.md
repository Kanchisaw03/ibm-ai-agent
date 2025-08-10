## Livelihood & Informal Economy Agent (FastAPI)

FastAPI app that analyzes urban livelihood inputs and the informal economy to recommend interventions, identify priority zones, predict risks, and render an interactive Folium map. Uses a local small LLM (FLAN‑T5) for text recommendations and scikit‑learn + pandas for numeric analysis.

### Features

- Local LLM (google/flan-t5-small) via LangChain for recommendations
- Numeric/statistical analysis (pandas + scikit-learn KMeans heuristics)
- Folium map with clustered markers and a bounding rectangle for priority areas
- Endpoints: `/health`, `/analyze_livelihood`, `/map/{map_id}`
- Minimal HTML/JS frontend at `/` (form + embedded map)
- Simple API key protection via `X-API-Key`

### Project Structure

```
livelihood_agent/
  app/
    main.py            # FastAPI app, routes, frontend
    config.py          # Pydantic settings (.env)
    schemas.py         # Pydantic request/response models + logging setup
    llm_agent.py       # FLAN-T5 pipeline, prompt builder, output parsing
    ml_predictor.py    # pandas + scikit-learn numeric analysis & risks
    map_generator.py   # Folium map generation, clustering, saving
  requirements.txt
  .env.example
  README.md
```

### Requirements

- Python 3.10+
- Windows/macOS/Linux

### Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell
pip install --upgrade pip
pip install -r livelihood_agent/requirements.txt
```

Copy environment file and edit:

```bash
copy livelihood_agent\.env.example .env   # PowerShell (Windows)
# or
cp livelihood_agent/.env.example .env      # macOS/Linux
```

`.env` variables:

```
API_KEY=change_me
```

### Run

```bash
uvicorn livelihood_agent.app.main:app --reload --host 0.0.0.0 --port 8001
```

Open:

- Swagger UI: `http://localhost:8001/docs`
- Minimal UI: `http://localhost:8001/`

### API

#### Health

```http
GET /health
```

#### Analyze Livelihood

```http
POST /analyze_livelihood
Content-Type: application/json
X-API-Key: <your_api_key>

{
  "location": "12.9716,77.5946",
  "population": 1200000,
  "income_distribution": {"low": 0.45, "mid": 0.4, "high": 0.15},
  "key_industries": ["retail", "services", "manufacturing"],
  "informal_sector_size": 0.38,
  "skills_profile": ["carpentry", "tailoring", "food services"],
  "policy_context": "City expansion plan prioritizes micro-enterprises and street vending regulation."
}
```

Response:

```json
{
  "recommendation": "...",
  "rationale": "...",
  "priority_zones": ["12.97,77.59", "12.98,77.60"],
  "predicted_risks": ["High dependency on informal sector"],
  "map_url": "/map/livelihood_..."
}
```

#### Map

```http
GET /map/{map_id}
```

Returns the generated Folium map as HTML.

### Frontend

- Minimal single-page form at `/` to submit inputs and render results.
- If `priority_zones` contains `"lat,lon"` strings, an interactive map iframe is embedded.

### Notes

- First run will download `google/flan-t5-small` weights; allow a couple of minutes.
- CPU inference works out of the box. For GPU, install a CUDA-enabled `torch` build.
- `map_generator.py` saves HTML maps under `livelihood_agent/output_maps/` (auto-created).
- Keep your `API_KEY` secret; required in the `X-API-Key` header for `/analyze_livelihood`.

### License

Apache-2.0
