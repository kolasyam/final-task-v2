# ADR-001: Finetuning Strategy for Sales Classification

## Status: Accepted

## Context

The project needs to classify sales representative field notes into 5 categories. The initial approach used an Ollama Modelfile with embedded few-shot examples, which was incorrectly labeled as "fine-tuning."

## Decision

We use a **hybrid three-tier approach**:

1. **QLoRA** (primary) — Real parameter-efficient fine-tuning on GPU
2. **Ollama Modelfile** (secondary) — Prompt engineering with few-shot examples
3. **Scikit-learn TF-IDF + RF** (fallback) — Fast local classification

## Rationale

### Why QLoRA as primary?
- Real weight-level fine-tuning produces the best accuracy
- 4-bit quantization fits gemma:2b in 8GB VRAM
- LoRA trains only ~1-2% of parameters (fast, memory-efficient)
- Adapter weights can be merged and exported to Ollama

### Why keep Ollama Modelfile?
- Zero GPU requirement for setup
- Works as a reliable fallback
- Easy to update with new examples
- Good enough for many use cases

### Why keep sklearn fallback?
- Sub-millisecond inference
- No LLM runtime required
- Works on any machine

## Consequences

- **Positive**: Best accuracy with QLoRA, graceful degradation to simpler methods
- **Positive**: No single point of failure
- **Negative**: More complex training pipeline
- **Negative**: QLoRA requires GPU and HuggingFace ecosystem

## Alternatives Considered

| Method | Why Rejected |
|--------|-------------|
| Full fine-tuning | Requires 16GB+ VRAM, overkill for 5-class classification |
| Embedding-based | Two-step process adds latency, marginal improvement over TF-IDF |
| GGUF + llama.cpp | Less ecosystem support, more complex conversion |
| Pure prompt engineering | Insufficient accuracy for production use |
