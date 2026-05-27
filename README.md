# Sales Intelligence Extractor

**Enterprise-Grade AI-Powered Sales Representative Intelligence Extraction System**

Classifies sales representative field notes into issue categories using a **fine-tuned gemma:2b model** trained on your dataset, with scikit-learn TF-IDF + RandomForest as fallback.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│              (User Interface - Port 8501)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend                            │
│              (API Layer - Port 8000)                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │ /predict  │  │/predict/ │  │ /health   │  │/categories│  │
│  │           │  │  batch   │  │           │  │           │  │
│  └─────┬────┘  └────┬─────┘  └───────────┘  └───────────┘  │
└────────┼────────────┼───────────────────────────────────────┘
         │            │
┌────────▼────────────▼───────────────────────────────────────┐
│              SalesNotePredictor (Orchestrator)               │
│                                                              │
│  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │  OllamaClient           │  │  SklearnClassifier        │  │
│  │  gemma-sales-intel ★    │  │  (TF-IDF + RandomForest)  │  │
│  │  (fine-tuned model)     │  │  Fallback backend         │  │
│  │  OR gemma:2b (base)     │  │                           │  │
│  └───────────┬────────────┘  └──────────┬────────────────┘  │
│              │                           │                    │
│  ┌───────────▼───────────────────────────▼────────────────┐  │
│  │              TextPreprocessor                           │  │
│  │  (lowercase, punctuation, abbreviation expansion)      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           PredictionStorage (CSV + JSONL)               │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────────┐
│                    MLflow Tracking                            │
│              (Experiment Logging - ./mlruns)                  │
└──────────────────────────────────────────────────────────────┘

★ = fine-tuned on your 500-record sales dataset
```

## Issue Categories

The system classifies sales notes into **5 categories**:

| # | Category | Description |
|---|----------|-------------|
| 1 | `supply_chain_delay` | Stock shortages, delivery delays, replenishment issues |
| 2 | `retailer_dissatisfaction` | Complaints, unhappy retailers, service issues |
| 3 | `pricing_conflict` | Price disputes, margin concerns, discount conflicts |
| 4 | `competitor_pressure` | Competitor actions, market share threats |
| 5 | `demand_spike` | Unexpected demand surges, stockouts from high volume |

---

## End-to-End Setup Guide

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- GPU recommended (for fast LLM inference)
- Your dataset at the configured path

---

### Step 1: Install Python Dependencies

```bash
cd "C:\Users\syamm\OneDrive\Desktop\Final Task\sales-intelligence"

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

**Required packages** (all VPN-compatible, no internet needed after install):
`fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `ollama`, `requests`, `pydantic`, `python-dotenv`, `matplotlib-inline`, `openpyxl`, `pytest`, `httpx`

---

### Step 2: Pull the Base Ollama Model

```bash
ollama pull gemma:2b
```

Verify it works:
```bash
ollama run gemma:2b "Hello, how are you?"
```

---

### Step 3: Configure Environment

```bash
copy .env.example .env
```

Edit `.env` and set your paths:
```env
DATASET_PATH=C:\Users\syamm\OneDrive\Desktop\final_standardized_sales_dataset.xlsx
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=gemma:2b
API_PORT=8000
STREAMLIT_PORT=8501
```

---

### Step 4: Fine-Tune the LLM on Your Dataset ⭐

This is the **most important step**. It creates a custom model `gemma-sales-intel` that is specialized for your sales data.

```bash
# Generate the Modelfile from your 500-record dataset
python -m training.generate_modelfile
```

This creates `training/saved_model/Modelfile` containing:
- Category definitions with descriptions
- 30 few-shot examples (6 per category) from your actual data
- Optimized inference parameters

Then create the fine-tuned Ollama model:
```bash
ollama create gemma-sales-intel -f training/saved_model/Modelfile
```

Verify the model was created:
```bash
ollama list
# Should show both 'gemma:2b' and 'gemma-sales-intel'
```

**What this does:** The `ollama create` command builds a new model based on gemma:2b with your sales classification knowledge baked into the system prompt. The model learns your 5 categories from your actual 500 sales notes.

> **Note:** The fine-tuning runs entirely on your local GPU via Ollama. No internet required. The process takes 2-10 minutes depending on your GPU.

---

### Step 5: Train the Scikit-Learn Fallback (Optional but Recommended)

```bash
python -m training.train
```

This trains a TF-IDF + RandomForest classifier on your dataset and saves artifacts to `training/saved_model/`. This serves as a fast fallback if Ollama is unavailable.

---

### Step 6: Start the FastAPI Backend

```bash
python -m app.main
```

The API starts at `http://localhost:8000`

On startup, it will show which model is being used:
```
✅ Using FINE-TUNED model: gemma-sales-intel
```

Or if fine-tuned model is not found:
```
ℹ️  Using base model: gemma:2b (fine-tune for better accuracy)
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check with model status |
| `GET` | `/api/v1/categories` | List supported categories |
| `GET` | `/api/v1/status` | Full system status |
| `POST` | `/api/v1/predict` | Classify a single note |
| `POST` | `/api/v1/predict/batch` | Classify multiple notes |

Test the API:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"rep_note": "Retailer reported stock running out of fast movers"}'
```

Swagger docs: `http://localhost:8000/docs`

---

### Step 7: Launch the Streamlit Dashboard

Open a **new terminal**:

```bash
venv\Scripts\activate
streamlit run streamlit_app/dashboard.py
```

Dashboard available at `http://localhost:8501`

The dashboard shows:
- **System Status** sidebar with fine-tune status
- **Prediction panel** — enter notes and get classified
- **Sample notes** — quick-test buttons for each category
- **Analytics** — category distribution and prediction history

---

## How Fine-Tuning Works

```
Your 500-record Excel dataset
         │
         ▼
┌─────────────────────────────┐
│  generate_modelfile.py      │
│  • Loads all 500 records    │
│  • Selects 30 best examples │
│    (6 per category)         │
│  • Builds system prompt     │
│    with category defs       │
│  • Writes Modelfile         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  ollama create              │
│  gemma-sales-intel          │
│  -f Modelfile               │
│                             │
│  Creates new Ollama model   │
│  specialized for your data  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  OllamaClient auto-detects  │
│  gemma-sales-intel and      │
│  uses it for all predictions│
└─────────────────────────────┘
```

The fine-tuned model `gemma-sales-intel` is used automatically. If it's not available, the system falls back to base `gemma:2b`, then to sklearn.

---

## Running Tests

```bash
pytest tests/ -v
```

50 tests covering API routes, prediction service, and text preprocessing.

---

## MLflow Tracking

View training experiments:

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser.

---

## Offline / VPN Installation

### On a machine WITH internet:

```bash
# Download all packages
pip download -r requirements.txt -d ./packages/

# Copy the entire project + packages folder to VPN machine
```

### On the VPN machine:

```bash
# Install from local packages (no internet needed)
pip install --no-index --find-links=./packages/ -r requirements.txt

# Ollama should be pre-installed on VPN host
# gemma:2b should be pre-pulled
ollama list    # verify gemma:2b is available

# Then follow Steps 3-7 above
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_PATH` | `...\final_standardized_sales_dataset.xlsx` | Path to Excel dataset |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL_NAME` | `gemma:2b` | Base Ollama model |
| `FINETUNED_MODEL_NAME` | `gemma-sales-intel` | Fine-tuned model name |
| `OLLAMA_TIMEOUT` | `120` | Ollama request timeout (seconds) |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI listen port |
| `STREAMLIT_PORT` | `8501` | Streamlit port |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow backend |
| `N_ESTIMATORS` | `200` | RandomForest tree count |
| `MAX_FEATURES` | `5000` | TF-IDF max features |
| `TEST_SIZE` | `0.2` | Test split fraction |
| `RANDOM_SEED` | `42` | Random seed |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `gemma:2b not found` | Run `ollama pull gemma:2b` |
| `gemma-sales-intel not found` | Run fine-tuning: `python -m training.generate_modelfile` then `ollama create gemma-sales-intel -f training/saved_model/Modelfile` |
| `Ollama connection refused` | Start Ollama: `ollama serve` |
| `sklearn artifacts not found` | Run: `python -m training.train` |
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| `Port already in use` | Change `API_PORT` or `STREAMLIT_PORT` in `.env` |

---

## License

Enterprise internal use only.
