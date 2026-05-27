# End-to-End Project Workflow

```mermaid
graph TD
    A["🚀 START: Local Machine<br/>(No VPN)"] --> B["1️⃣ Setup Local Environment<br/>• Python 3.10+<br/>• Create venv<br/>• pip install -r requirements.txt"]
    
    B --> C["2️⃣ Verify Dataset<br/>• python check_environment.py<br/>• Confirm 500 records<br/>• Check columns: rep_note, issue_category"]
    
    C --> D["✅ Local Setup Complete<br/>Ready for VPN connection"]
    
    D -->|Connect to VPN with GPU| E["🖥️ VPN Machine with GPU<br/>(8GB+ VRAM)"]
    
    E --> F["3️⃣ Verify Ollama on VPN<br/>• ollama list<br/>• Confirm gemma:2b installed<br/>• ollama serve (start server)"]
    
    F --> G["4️⃣ Configure .env File<br/>• OLLAMA_BASE_URL<br/>• DATASET_PATH<br/>• QLORA settings"]
    
    G --> H["5️⃣ Prepare Dataset<br/>python -m training.prepare_dataset<br/>↓<br/>Creates train.jsonl & test.jsonl<br/>(80/20 split)"]
    
    H --> I["6️⃣ Train Scikit-learn Fallback<br/>python -m training.train<br/>↓<br/>TF-IDF Vectorizer +<br/>RandomForest Classifier<br/>⏱️ ~5 minutes"]
    
    I --> J{"Want<br/>Fine-tuning?"}
    
    J -->|YES: Best Accuracy| K["7A️⃣ QLoRA Fine-tuning<br/>python -m training.finetune_qlora<br/>↓<br/>✓ 4-bit quantization<br/>✓ LoRA adapters<br/>✓ 3 epochs training<br/>✓ GPU required<br/>⏱️ ~45 minutes"]
    
    J -->|NO: Quick Setup| L["7B️⃣ Prompt Engineering Only<br/>python -m training.generate_modelfile<br/>↓<br/>✓ Few-shot examples<br/>✓ No GPU needed<br/>⏱️ ~2 minutes"]
    
    K --> M["8️⃣ Evaluate QLoRA<br/>python -m training.evaluate_qlora<br/>↓<br/>• F1-score<br/>• Confusion matrix<br/>• Sample predictions"]
    
    L --> N["8️⃣ Create Ollama Model<br/>ollama create gemma-sales-intel \<br/>  -f training/saved_model/Modelfile"]
    
    M --> O["9️⃣ Convert to Ollama<br/>python -m training.convert_to_ollama<br/>↓<br/>ollama create gemma-sales-intel-qlora \<br/>  -f training/saved_model/Modelfile.qlora"]
    
    N --> P["🎯 Models Ready"]
    O --> P
    
    P --> Q["🚀 Start Backend<br/>Terminal 1:<br/>python -m app.main<br/>↓<br/>FastAPI running on :8000"]
    
    Q --> R["🎨 Start Dashboard<br/>Terminal 2:<br/>streamlit run streamlit_app/dashboard.py<br/>↓<br/>Dashboard running on :8501"]
    
    R --> S["📊 System is LIVE"]
    
    S --> T["✨ Usage Options"]
    
    T --> U["Option 1: Web UI<br/>http://localhost:8501<br/>• Enter sales notes<br/>• Click Predict<br/>• View results"]
    
    T --> V["Option 2: API<br/>curl -X POST \<br/>  http://localhost:8000/api/v1/predict \<br/>  -d '{rep_note: ...}'"]
    
    T --> W["Option 3: Batch<br/>POST /api/v1/predict/batch<br/>• Process multiple notes<br/>• Get predictions array"]
    
    U --> X["📈 Monitor"]
    V --> X
    W --> X
    
    X --> Y["MLflow Tracking<br/>mlflow ui<br/>→ http://localhost:5000<br/>• View experiments<br/>• Compare metrics<br/>• Track artifacts"]
    
    Y --> Z["📊 View Predictions<br/>data/predictions.csv<br/>• Category distribution<br/>• Confidence scores<br/>• Inference latency"]
    
    Z --> AA["✅ Project Complete!<br/>Production-ready<br/>Sales Intelligence System"]
    
    style A fill:#90EE90
    style E fill:#FFB6C1
    style P fill:#87CEEB
    style S fill:#DDA0DD
    style AA fill:#FFD700
