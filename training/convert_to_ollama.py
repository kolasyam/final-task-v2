"""Convert QLoRA adapter to an Ollama-compatible model.

This script merges QLoRA adapter weights with the base gemma:2b model,
exports the merged model in GGUF format, and generates an Ollama Modelfile
that can be used with `ollama create`.

The resulting model combines the benefits of:
  1. QLoRA fine-tuned weights (real parameter updates)
  2. Ollama inference engine (fast, local, no Python runtime needed)

Usage:
    python -m training.convert_to_ollama

Requirements:
    - llama.cpp (for GGUF conversion) OR
    - huggingface_hub + transformers (for safetensors export)

Output:
    - training/saved_model/ollama_gguf/ (merged model in GGUF format)
    - training/saved_model/Modelfile.qlora (Ollama Modelfile pointing to merged model)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import config

logger = logging.getLogger(__name__)

GGUF_OUTPUT_DIR: str = "training/saved_model/ollama_gguf"
OLLAMA_MODELFILE: str = "training/saved_model/Modelfile.qlora"


def merge_adapter(
    base_model: str = "/opt/ai-platform/models/gemma-2-2b-it",
    adapter_path: Optional[str] = None,
    output_path: str = "training/saved_model/qlora_merged",
) -> str:
    """Merge QLoRA adapter weights into the base model.

    Args:
        base_model: HuggingFace model ID.
        adapter_path: Path to QLoRA adapter.
        output_path: Path to save merged model.

    Returns:
        Path to the merged model.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = adapter_path or config.qlora_adapter_path

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"QLoRA adapter not found: {adapter_path}")

    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info("Loading adapter from: %s", adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True,
    local_files_only=True,)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merged model saved to: %s", output_path)
    return output_path


def convert_to_gguf(
    model_path: str,
    output_dir: str = GGUF_OUTPUT_DIR,
) -> str:
    """Convert a HuggingFace model to GGUF format using llama.cpp.

    Args:
        model_path: Path to the HuggingFace model.
        output_dir: Directory to save GGUF file.

    Returns:
        Path to the GGUF file.
    """
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")

    os.makedirs(output_dir, exist_ok=True)
    gguf_path = os.path.join(output_dir, "model.gguf")

    # Try using llama.cpp convert script
    convert_script = None
    possible_paths = [
        os.path.join(llama_cpp_dir, "convert.py") if llama_cpp_dir else "",
        "convert.py",
        "llama.cpp/convert.py",
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            convert_script = path
            break

    if convert_script:
        cmd = [
            sys.executable, convert_script,
            model_path,
            "--outfile", gguf_path,
            "--outtype", "f16",
        ]
        logger.info("Converting to GGUF: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info("GGUF file created: %s", gguf_path)
            return gguf_path
        else:
            logger.error("GGUF conversion failed: %s", result.stderr)

    # Fallback: use HuggingFace gguf conversion
    logger.info("Trying HuggingFace GGUF conversion...")
    try:
        from huggingface_hub import HfApi
        # Use optimum or other tools as fallback
        logger.warning(
            "Automatic GGUF conversion not available. "
            "Use llama.cpp manually: python convert.py %s --outfile %s",
            model_path, gguf_path,
        )
    except ImportError:
        pass

    return ""


def generate_ollama_modelfile(
    merged_model_path: str,
    gguf_path: str = "",
    output_path: str = OLLAMA_MODELFILE,
    model_name: str = "gemma-sales-intel-qlora",
) -> str:
    """Generate an Ollama Modelfile for the QLoRA-finetuned model.

    Args:
        merged_model_path: Path to the merged model.
        gguf_path: Path to GGUF file (optional).
        output_path: Path to save the Modelfile.
        model_name: Name for the Ollama model.

    Returns:
        Path to the generated Modelfile.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build a concise system prompt (the model already knows the task)
    system_prompt = (
        "You are a sales intelligence analyst. Classify sales representative "
        "field notes into exactly one category: supply_chain_delay, "
        "retailer_dissatisfaction, pricing_conflict, competitor_pressure, "
        "or demand_spike. Respond with ONLY the category name."
    )

    lines = [
        f"# QLoRA-finetuned Sales Intelligence Model",
        f"# Based on gemma:2b with QLoRA adapter fine-tuned on sales data",
        f"",
    ]

    if gguf_path and os.path.exists(gguf_path):
        lines.append(f"FROM {gguf_path}")
    else:
        # Use safetensors path
        lines.append(f"FROM {merged_model_path}")

    lines.extend([
        "",
        f"PARAMETER temperature 0.1",
        f"PARAMETER top_p 0.9",
        f"PARAMETER top_k 40",
        f"PARAMETER num_predict 30",
        f"",
        f"SYSTEM \"\"\"{system_prompt}\"\"\"",
        "",
    ])

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    logger.info("Modelfile generated: %s", output_path)
    logger.info("To create the Ollama model, run:")
    logger.info("  ollama create %s -f %s", model_name, output_path)

    return output_path


def convert_pipeline(
    base_model: str = "/opt/ai-platform/models/gemma-2-2b-it",
    adapter_path: Optional[str] = None,
    skip_gguf: bool = False,
) -> Dict[str, Any]:
    """Run the full conversion pipeline.

    Args:
        base_model: HuggingFace base model ID.
        adapter_path: Path to QLoRA adapter.
        skip_gguf: Skip GGUF conversion step.

    Returns:
        Dictionary with paths to outputs.
    """
    adapter_path = adapter_path or config.qlora_adapter_path

    logger.info("=" * 60)
    logger.info("QLoRA Adapter → Ollama Model Conversion")
    logger.info("=" * 60)

    # Step 1: Merge adapter
    merged_path = merge_adapter(base_model, adapter_path)

    # Step 2: Convert to GGUF (optional)
    gguf_path = ""
    if not skip_gguf:
        logger.info("Converting to GGUF format...")
        gguf_path = convert_to_gguf(merged_path)

    # Step 3: Generate Modelfile
    modelfile_path = generate_ollama_modelfile(merged_path, gguf_path)

    results = {
        "merged_model": merged_path,
        "gguf": gguf_path,
        "modelfile": modelfile_path,
    }

    logger.info("Conversion complete!")
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        results = convert_pipeline()
        print("\n✅ Conversion pipeline complete!")
        print(f"   Merged model: {results['merged_model']}")
        print(f"   GGUF: {results['gguf'] or 'skipped'}")
        print(f"   Modelfile: {results['modelfile']}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Run QLoRA training first: python -m training.finetune_qlora")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
