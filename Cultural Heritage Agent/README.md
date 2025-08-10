## Participatory Governance & Cultural Heritage Agent (FastAPI)

Production-ready FastAPI app for analyzing citizen feedback, mapping grievances to governance actions, and highlighting cultural heritage concerns. Built with a clean, modular structure and a hybrid rule-based + local LLM approach.

### Features

- Analyze feedback: text cleaning, sentiment (HF model), grievance extraction
- Local LLM (FLAN-T5) generates concise recommendations
- Rule-based categorization and merge logic with LLM outputs
- Folium map with markers and popups; maps served as static HTML
- Endpoints: `/health`, `/analyze_feedback`, `/load_sample_data`
- Logging of each API call to `logs/app.log`

### Project Structure

```
app/
  main.py                 # FastAPI entrypoint
  config.py               # Pydantic settings (.env)
  models.py               # Pydantic request/response schemas
  nlp_processor.py        # Cleaning, sentiment, grievance extraction
  governance_logic.py     # Rule-based mapping & fallback recommendations
  llm_agent.py            # FLAN-T5 pipeline and prompt/parse
  map_gui.py              # Folium map generation/saving
data/
  sample_feedback.json    # Example grievances (>=5)
models/
  .gitkeep
static/
  maps/
    .gitkeep
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
pip install -r requirements.txt
```

Copy environment file and edit if needed:

```bash
cp .env.example .env
```

`.env` variables:

```
# Optional; public models do not require a token
HUGGINGFACE_TOKEN=

# Override models if desired
HF_MODEL_NAME=google/flan-t5-small
SENTIMENT_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
```

### Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000/docs` for Swagger UI.

### API

#### Health

```http
GET /health
```

#### Analyze Feedback

```http
POST /analyze_feedback
Content-Type: application/json

{
  "citizen_id": "CIT001",
  "feedback_text": "There are frequent power cuts near the old fort; the museum area is very dark at night.",
  "location_lat": 12.9716,
  "location_lon": 77.5946,
  "timestamp": "2025-07-21T12:34:56Z"
}
```

Response contains sentiment, grievance category, recommendation, and a `map_url` to the generated HTML map.

#### Load Sample Data

```http
GET /load_sample_data
```

Processes `data/sample_feedback.json` and returns a combined map URL with all sample markers.

### Notes

- First run will download models to your HF cache; this can take a few minutes.
- CPU inference works out-of-the-box. For GPU, install the appropriate CUDA-enabled `torch`.
- If outbound access is restricted, place your HF models under `models/` and set environment variables to point to local paths.

### License

Apache-2.0
