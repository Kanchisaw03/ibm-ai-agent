## Housing Risk Assessment Agent (FastAPI)

FastAPI app that evaluates housing risk using a tiny local ML model (MinMaxScaler + KMeans) and generates concise recommendations with a local small LLM (FLAN‑T5 via HuggingFacePipeline). Produces an interactive Folium map and returns a JSON response.

### Features

- Local LLM: `google/flan-t5-small` through LangChain's `HuggingFacePipeline`
- Local ML: scaler + KMeans clustering; heuristic risk labels (low/medium/high)
- Folium maps saved under `housing_agent/output_maps/`
- Endpoints: `/health`, `/analyze_housing`, `/map/{filename}`
- No authentication required (local/offline use)

### Structure

```
housing_agent/
  app/
    main.py            # FastAPI app, routes, minimal frontend
    config.py          # Pydantic settings (.env)
    schemas.py         # Pydantic models + logging setup
    ml_predictor.py    # Scaler + KMeans, risk labels and tags
    llm_agent.py       # FLAN‑T5 prompt + output parsing
    map_generator.py   # Folium map creation
  requirements.txt
  ENV_SAMPLE.txt
  data/
    housing_dataset.csv
```

### Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell
pip install --upgrade pip
pip install -r housing_agent/requirements.txt
```

Copy environment file and edit:

```bash
copy housing_agent\ENV_SAMPLE.txt .env   # Windows PowerShell
# or
cp housing_agent/ENV_SAMPLE.txt .env      # macOS/Linux
```

`.env` variables:

```
MODEL_DIR=
```

### Run

```bash
uvicorn housing_agent.app.main:app --reload --host 0.0.0.0 --port 8003
```

Open:

- Swagger UI: `http://localhost:8003/docs`
- Minimal UI: `http://localhost:8003/`

### API

#### Analyze Housing

```http
POST /analyze_housing
Content-Type: application/json

{
  "location_name": "Koramangala",
  "latitude": 12.9352,
  "longitude": 77.6245,
  "housing_condition_score": 45,
  "population_density": 18000,
  "low_income_household_percentage": 62,
  "access_to_services_score": 28
}
```

Response:

```json
{
  "recommendation": "...",
  "rationale": "...",
  "priority_zones": ["12.94,77.63", "12.93,77.62"],
  "predicted_risks": ["Overcrowding", "Poor access to services"],
  "map_url": "/map/housing_..."
}
```

### Notes

- The first run will download `google/flan-t5-small` weights.
- Runs offline after models are cached locally.
- The included `ml_predictor.py` will initialize and persist a tiny KMeans model if none exists.
