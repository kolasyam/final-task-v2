# Deployment Guide

## Linux VM with GPU (Recommended for Production)

### Prerequisites
- Ubuntu 20.04+ / CentOS 8+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+ and cuDNN installed

### Step 1: Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull gemma:2b
```

### Step 2: Set Up Python Environment
```bash
cd /path/to/sales-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
cp .env.example .env
# Edit .env and set:
# - DATASET_PATH (path to your Excel dataset)
# - OLLAMA_BASE_URL (default: http://localhost:11434)
# - API_KEY (set a strong random key)
# - CORS_ORIGINS (set your dashboard URL)
```

### Step 4: Train Models
```bash
# Option A: Prompt engineering (no GPU needed)
python -m training.generate_modelfile
ollama create gemma-sales-intel -f training/saved_model/Modelfile

# Option B: QLoRA fine-tuning (GPU required, better accuracy)
python -m training.finetune_qlora

# Train sklearn fallback
python -m training.train
```

### Step 5: Start Services
```bash
# API (port 8000)
uvicorn app.main --host 0.0.0.0 --port 8000 --workers 4

# Dashboard (port 8501)
streamlit run streamlit_app/dashboard.py --server.port 8501
```

### Step 6: Verify
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Prediction test
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"rep_note": "Retailer reported stock running out"}'
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "app.main", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sales-intelligence .
docker run -p 8000:8000 -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e API_KEY=your-secret-key \
  sales-intelligence
```

## Systemd Service (Linux)

```ini
# /etc/systemd/system/sales-intelligence.service
[Unit]
Description=Sales Intelligence API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/sales-intelligence
Environment=PATH=/path/to/sales-intelligence/venv/bin
ExecStart=/path/to/sales-intelligence/venv/bin/uvicorn app.main --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable sales-intelligence
sudo systemctl start sales-intelligence
sudo systemctl status sales-intelligence
```
