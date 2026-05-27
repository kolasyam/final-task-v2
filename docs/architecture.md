# Sales Intelligence — Architecture Documentation

## System Overview

The Sales Intelligence Extractor classifies sales representative field notes into 5 issue categories using a tiered inference strategy:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│              (User Interface - Port 8501)                    │
│              http://localhost:8501                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST (JSON)
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend                            │
│              (API Layer - Port 8000)                         │
│              http://localhost:8000                           │
│                                                              │
│  Middleware Stack (in order):                                │
│  1. CORS (env-configurable origins)                          │
│  2. Rate Limiting (token bucket, 60 RPM default)             │
│  3. Correlation IDs (X-Correlation-ID header)                │
│  4. API Key Auth (X-API-Key header, optional)                │
│                                                              │
│  Endpoints:                                                  │
│  GET  /api/v1/health      — Health check                     │
│  GET  /api/v1/categories  — List categories                  │
│  GET  /api/v1/status      — Full system status               │
│  POST /api/v1/predict     — Single note classification       │
│  POST /api/v1/predict/batch — Batch classification           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              SalesNotePredictor (Orchestrator)               │
│                                                              │
│  Inference Priority:                                         │
│  1. Prompt-engineered Ollama model (gemma-sales-intel)       │
│     → Ollama Modelfile with embedded few-shot examples       │
│     → Best accuracy without GPU training                     │
│                                                              │
│  2. Base Ollama model (gemma:2b)                             │
│     → Zero-shot classification with full prompt              │
│     → Good generalization, no training needed                │
│                                                              │
│  3. Scikit-learn TF-IDF + RandomForest                       │
│     → Fast local fallback (~1ms inference)                   │
│     → No LLM required                                       │
│                                                              │
│  Components:                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  OllamaClient                                         │    │
│  │  • Async HTTP communication with Ollama server        │    │
│  │  • Circuit breaker (5 failures → 30s cooldown)        │    │
│  │  • Auto-detects best available model                  │    │
│  │  • Category extraction with fuzzy matching            │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  SklearnClassifier                                    │    │
│  │  • Loads TF-IDF vectorizer + RandomForest artifacts   │    │
│  │  • Fast local inference                               │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  TextPreprocessor                                     │    │
│  │  • Lowercase, punctuation cleanup                     │    │
│  │  • 134 sales abbreviation expansions                  │    │
│  │  • Whitespace normalization                           │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  PredictionStorage                                    │    │
│  │  • Dual-write to CSV + JSONL                          │    │
│  │  • Query history and category counts                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    MLflow Tracking                            │
│              (Experiment Logging - ./mlruns)                  │
│  • Training parameters and metrics                           │
│  • Model artifact versioning                                 │
│  • Cross-validation scores                                    │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. User Input (sales note text)
   ↓
2. API Validation (Pydantic schema, max 1000 chars)
   ↓
3. Rate Limiting Check (token bucket)
   ↓
4. Correlation ID Assignment (for tracing)
   ↓
5. Text Preprocessing
   • Lowercase conversion
   • Punctuation cleanup
   • Abbreviation expansion (134 sales abbreviations)
   • Whitespace normalization
   ↓
6. Model Selection (tiered)
   a. Prompt-engineered Ollama model (if available)
   b. Base Ollama model (if Ollama reachable)
   c. Scikit-learn fallback (if artifacts exist)
   ↓
7. Category Extraction
   • Direct match → return
   • Substring match → return
   • Keyword scoring → best match
   • No match → fallback or default
   ↓
8. Confidence Estimation
   • Ollama: based on output clarity (0.60-0.95)
   • Sklearn: based on prediction probability
   ↓
9. Storage (CSV + JSONL)
   ↓
10. Response (category, confidence, method, latency, reasoning)
```

## Model Training Pipeline

```
1. Dataset Loading (Excel → pandas DataFrame)
   ↓
2. Validation & Filtering
   • Remove empty notes
   • Filter unknown categories
   • Detect duplicates (data leakage warning)
   ↓
3. Category Mapping (UPPERCASE → normalized names)
   ↓
4. Stratified Train/Test Split (80/20)
   ↓
5. Cross-Validation (5-fold, detects overfitting)
   ↓
6. Training (TF-IDF + RandomForest)
   ↓
7. Evaluation (accuracy, precision, recall, F1)
   ↓
8. Artifact Persistence (vectorizer, classifier, label encoder)
   ↓
9. MLflow Logging (params, metrics, artifacts)
```

## Finetuning Methods

### Method 1: Prompt Engineering (Ollama Modelfile)
- **File**: `training/generate_modelfile.py`
- **What**: Embeds few-shot examples in system prompt
- **GPU Required**: No
- **Weight Updates**: None (prompt only)
- **Accuracy**: Moderate
- **Use Case**: Quick baseline, no GPU available

### Method 2: QLoRA (Real Parameter Fine-Tuning) ⭐
- **File**: `training/finetune_qlora.py`
- **What**: 4-bit quantized LoRA adapter training
- **GPU Required**: Yes (8GB+ VRAM)
- **Weight Updates**: ~1-2% of parameters (LoRA adapters)
- **Accuracy**: Best
- **Use Case**: Production deployment with GPU

### Method 3: Scikit-Learn Fallback
- **File**: `training/train.py`
- **What**: TF-IDF + RandomForest classifier
- **GPU Required**: No
- **Weight Updates**: Full model training
- **Accuracy**: Good for template data, may overfit
- **Use Case**: Fast fallback, no LLM available

## Security Architecture

| Layer | Mechanism | Configuration |
|-------|-----------|---------------|
| CORS | Origin whitelist | `CORS_ORIGINS` env var |
| Rate Limiting | Token bucket | `RATE_LIMIT_RPM` env var (default: 60) |
| Authentication | API key in header | `API_KEY` env var, `X-API-Key` header |
| Input Validation | Pydantic schemas | Max 1000 chars per note, max 50 per batch |
| Correlation ID | UUID per request | `X-Correlation-ID` header |

## Scalability Considerations

- **Stateless API**: No server-side session state
- **Singleton Predictor**: Loaded once, reused across requests
- **Circuit Breaker**: Prevents cascading failures
- **Rate Limiting**: Protects against abuse
- **Async-Ready**: FastAPI supports async handlers (Ollama client uses sync requests — upgrade to httpx for full async)

## Deployment Options

### Local Development
```bash
python -m app.main          # API on port 8000
streamlit run streamlit_app/dashboard.py  # Dashboard on port 8501
```

### Docker (recommended for production)
See `docs/deployment.md`

### Linux VM with GPU
1. Install Ollama on the VM
2. Pull gemma:2b: `ollama pull gemma:2b`
3. Run prompt engineering: `python -m training.generate_modelfile && ollama create gemma-sales-intel -f training/saved_model/Modelfile`
4. (Optional) Run QLoRA: `python -m training.finetune_qlora`
5. Start API: `python -m app.main`
