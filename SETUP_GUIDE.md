# Sales Intelligence Extractor — Complete End-to-End Setup Guide

## Project Overview

This is a hybrid AI system that classifies sales representative notes into 5 issue categories:
- 🧠 **Primary**: QLoRA fine-tuned gemma:2b (GPU required, best accuracy)
- 🔧 **Secondary**: Ollama Modelfile with prompt engineering (fallback)
- ⚡ **Fallback**: Scikit-learn TF-IDF + RandomForest (no GPU needed)

**Architecture**: Streamlit Dashboard → FastAPI Backend → OllamaClient + SklearnClassifier

---

## Prerequisites

### Required
- Python 3.10+
- VPN connection with GPU access
- Ollama installed on VPN with `gemma:2b` model
- Dataset: `data/final_standardized_sales_dataset.xlsx` (500 records)

### GPU Requirements
- NVIDIA GPU with **8GB+ VRAM** (for QLoRA fine-tuning)
- CUDA 11.8+ and cuDNN installed on VPN machine
- PyTorch with CUDA support

---

## Phase 1: Local Setup (Without VPN)

### Step 1.1: Clone/Navigate to Project

```bash
cd "C:\Users\syamm\OneDrive\Desktop\Final Task\sales-intelligence"
```

### Step 1.2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Or activate existing venv
venv\Scripts\activate
```

### Step 1.3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi`, `uvicorn` (API server)
- `streamlit` (dashboard)
- `ollama`, `requests` (LLM integration)
- `scikit-learn`, `pandas` (ML models)
- `peft`, `trl`, `bitsandbytes`, `transformers` (QLoRA fine-tuning)
- `mlflow` (experiment tracking)

### Step 1.4: Verify Dataset

```bash
python check_environment.py
```

**Expected output:**
- ✅ Dataset found at `data/final_standardized_sales_dataset.xlsx`
- ✅ 500+ records with `rep_note` and `issue_category` columns

---

## Phase 2: Connect to VPN with GPU

### Step 2.1: VPN Connection

Connect to your VPN that provides GPU access.

### Step 2.2: Verify Ollama on VPN

From the VPN machine, verify gemma:2b is installed:

```bash
ollama list
# Expected: gemma:2b             2.0 GB
```

If not installed:
```bash
ollama pull gemma:2b
ollama serve  # Start Ollama server on 0.0.0.0:11434
```

### Step 2.3: Configure .env File

Create `.env` file in the project root:

```bash
# .env

# === Ollama Configuration ===
OLLAMA_BASE_URL=http://localhost:11434          # Local if running on same machine
MODEL_NAME=gemma:2b
OLLAMA_TIMEOUT=120

# === Dataset ===
DATASET_PATH=data/final_standardized_sales_dataset.xlsx

# === API Server ===
API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=http://localhost:8000

# === Streamlit ===
STREAMLIT_PORT=8501

# === MLflow Tracking ===
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=sales-intelligence

# === QLoRA Fine-tuning ===
QLORA_EPOCHS=3
QLORA_LEARNING_RATE=2e-4
QLORA_BATCH_SIZE=4
QLORA_GRADIENT_ACCUMULATION_STEPS=1

# === Model Directories ===
MODEL_DIR=training/saved_model
QLORA_ADAPTER_PATH=training/saved_model/qlora_adapter
```

---

## Phase 3: Train Models (On VPN with GPU)

### Step 3.1: Prepare Dataset

```bash
python -m training.prepare_dataset
```

**Output:**
- `training/dataset/train.jsonl` (400 records)
- `training/dataset/test.jsonl` (100 records)

### Step 3.2: Train Scikit-learn Fallback Model

```bash
python -m training.train
```

**Output:**
- `training/saved_model/vectorizer.joblib` (TF-IDF)
- `training/saved_model/classifier.joblib` (RandomForest)
- `training/saved_model/label_encoder.joblib`
- Classification report in MLflow

**Time**: ~2-5 minutes

### Step 3.3: Train QLoRA Fine-tuned Model (GPU Required)

```bash
python -m training.finetune_qlora
```

**What it does:**
1. Downloads `google/gemma-2b` from HuggingFace
2. Applies 4-bit quantization to fit in GPU VRAM
3. Adds LoRA adapters to attention/MLP layers
4. Fine-tunes on your training dataset (3 epochs)
5. Saves adapter weights to `training/saved_model/qlora_adapter/`

**Requirements:**
- GPU with 8GB+ VRAM
- Internet connection (first run to download model)
- ~30-60 minutes depending on GPU

**Output:**
- `training/saved_model/qlora_adapter/adapter_config.json`
- `training/saved_model/qlora_adapter/adapter_model.bin`
- MLflow experiment tracking

**Expected Result:**
- Better accuracy than base model
- Each prediction ~2-5 seconds

### Step 3.4: Evaluate QLoRA Model

```bash
python -m training.evaluate_qlora
```

**Output:**
- F1-score comparison (base vs fine-tuned)
- Confusion matrix
- Sample predictions

### Step 3.5: Convert QLoRA to Ollama Format (Optional)

```bash
python -m training.convert_to_ollama
```

**What it does:**
1. Merges LoRA adapter with base gemma:2b
2. Converts to GGUF format (quantized)
3. Generates Ollama Modelfile

**Output:**
- `training/saved_model/ollama_gguf/` (merged model)
- `training/saved_model/Modelfile.qlora`

**Then create Ollama model:**
```bash
ollama create gemma-sales-intel-qlora -f training/saved_model/Modelfile.qlora
```

**Benefit**: Run fine-tuned model through Ollama (faster inference)

### Step 3.6: Alternative - Prompt Engineering Only (No GPU)

If QLoRA fails or you want faster setup:

```bash
python -m training.build_few_shot
python -m training.generate_modelfile
```

**Output:**
- `training/saved_model/Modelfile` (with few-shot examples)

**Create Ollama model:**
```bash
ollama create gemma-sales-intel -f training/saved_model/Modelfile
```

**Benefit**: Quick setup, no GPU needed, ~70% accuracy

---

## Phase 4: Run the System

### Step 4.1: Start FastAPI Backend

```bash
python -m app.main
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open.**

### Step 4.2: Open New Terminal & Start Streamlit Dashboard

```bash
cd "C:\Users\syamm\OneDrive\Desktop\Final Task\sales-intelligence"
venv\Scripts\activate
streamlit run streamlit_app/dashboard.py
```

**Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 4.3: Access the Dashboard

Open browser and navigate to:
- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/api/v1/health

---

## Phase 5: Use the System

### Using the Dashboard

1. **Enter Sales Note** in the text area
2. **Click "Predict Category"** button
3. **View Results**:
   - Issue Category
   - Confidence Score
   - Method Used (Ollama, sklearn, etc.)
   - Reasoning

### Using the API Directly

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"rep_note": "Retailer ran out of stock, customers unhappy"}'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"rep_note": "Note 1"},
    {"rep_note": "Note 2"},
    {"rep_note": "Note 3"}
  ]'
```

**Get Categories:**
```bash
curl http://localhost:8000/api/v1/categories
```

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

---

## Phase 6: Monitor & Optimize

### Check MLflow Experiments

```bash
mlflow ui
```

Navigate to http://localhost:5000 to see:
- Training parameters
- Accuracy metrics
- Cross-validation scores
- Model artifacts

### View Predictions

```python
import pandas as pd

# Load predictions made by the system
df = pd.read_csv("data/predictions.csv")
print(df.head())
print(df['issue_category'].value_counts())
```

### Test on Your Data

```bash
python -c "
import pandas as pd
import requests

# Load your notes
notes = ['Note 1', 'Note 2', 'Note 3']

# Batch predict
response = requests.post(
    'http://localhost:8000/api/v1/predict/batch',
    json=[{'rep_note': n} for n in notes]
)

results = response.json()
for r in results:
    print(f\"{r['rep_note'][:50]}... → {r['issue_category']}\")
"
```

---

## Inference Strategy (Auto-Selection)

The system automatically selects the best available model:

1. **Fine-tuned QLoRA Ollama model** (if exists)
   - Best accuracy (75-85%)
   - Requires GPU training only, inference fast

2. **Prompt-engineered Ollama model** (if exists)
   - Good accuracy (65-75%)
   - No GPU needed

3. **Base gemma:2b Ollama model**
   - Decent accuracy (60-70%)
   - Zero-shot classification

4. **Scikit-learn TF-IDF + RandomForest** (fallback)
   - Reasonable accuracy (65-75%)
   - Sub-millisecond inference
   - Works offline

---

## Troubleshooting

### Ollama Connection Error
```
ConnectionError: Failed to connect to Ollama at http://localhost:11434
```

**Fix:**
```bash
# On VPN machine, ensure Ollama is running:
ollama serve
```

### GPU Out of Memory During QLoRA
```
CUDA out of memory
```

**Fix:**
- Reduce `QLORA_BATCH_SIZE` in `.env` (try 2 or 1)
- Reduce `QLORA_GRADIENT_ACCUMULATION_STEPS`
- Free GPU memory: `nvidia-smi` and close other GPU apps

### Streamlit Connection Refused
```
Error: Could not connect to API at http://localhost:8000
```

**Fix:**
```bash
# Ensure FastAPI is running:
python -m app.main
```

### Model Not Found
```
Model 'gemma:2b' not found on Ollama server
```

**Fix:**
```bash
ollama pull gemma:2b
```

---

## Quick Start Commands (Copy-Paste)

### First Time Setup
```bash
cd "C:\Users\syamm\OneDrive\Desktop\Final Task\sales-intelligence"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### On VPN with GPU (Full Training)
```bash
python -m training.prepare_dataset
python -m training.train
python -m training.finetune_qlora      # GPU fine-tuning (~45 min)
python -m training.evaluate_qlora
```

### Run the System
**Terminal 1:**
```bash
python -m app.main
```

**Terminal 2:**
```bash
streamlit run streamlit_app/dashboard.py
```

### Monitor
```bash
mlflow ui  # Navigate to http://localhost:5000
```

---

## Project Structure

```
sales-intelligence/
├── training/
│   ├── train.py                    # Scikit-learn training
│   ├── finetune_qlora.py           # QLoRA fine-tuning (GPU)
│   ├── evaluate_qlora.py           # Evaluation
│   ├── convert_to_ollama.py        # Convert to GGUF
│   ├── generate_modelfile.py       # Prompt engineering
│   ├── dataset/
│   │   ├── train.jsonl             # Training data
│   │   └── test.jsonl              # Test data
│   └── saved_model/
│       ├── classifier.joblib       # sklearn model
│       ├── vectorizer.joblib       # TF-IDF
│       ├── qlora_adapter/          # QLoRA weights
│       └── Modelfile               # Ollama definition
├── app/
│   ├── main.py                     # FastAPI app
│   ├── config.py                   # Configuration
│   ├── api/routes.py               # API endpoints
│   └── services/
│       ├── ollama_client.py        # Ollama integration
│       ├── predictor.py            # Inference orchestrator
│       └── preprocessing.py        # Text processing
├── streamlit_app/
│   └── dashboard.py                # Web UI
├── data/
│   ├── final_standardized_sales_dataset.xlsx
│   ├── predictions.csv             # Prediction history
│   └── predictions.jsonl           # Structured logs
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   └── adr/
│       └── 001-finetuning-strategy.md
└── mlruns/                         # MLflow experiments
```

---

## Next Steps

1. ✅ **Local Setup**: Install dependencies, verify dataset
2. ✅ **VPN Connection**: Connect to GPU-enabled VPN
3. ✅ **Train Models**: Run scikit-learn + QLoRA training
4. ✅ **Deploy**: Start FastAPI + Streamlit
5. ✅ **Use**: Test with real sales notes
6. ✅ **Monitor**: View MLflow experiments and predictions
7. ✅ **Optimize**: Fine-tune hyperparameters based on results

---

## Support

- **Architecture Details**: See `docs/architecture.md`
- **Deployment Details**: See `docs/deployment.md`
- **Design Decisions**: See `docs/adr/001-finetuning-strategy.md`
- **API Docs**: http://localhost:8000/docs (when running)

---

## File Size Reference

After training:
- Base model (gemma:2b): ~2 GB
- QLoRA adapter: ~50 MB
- Sklearn model: ~2 MB
- GGUF conversion: ~1.4 GB (optional)

**Total**: ~3.5 GB (base model) + optional artifacts
