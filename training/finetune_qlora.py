"""QLoRA fine-tuning for gemma:2b on the sales classification dataset.

This module performs REAL parameter-efficient fine-tuning using:
  - 4-bit quantization (bitsandbytes) to fit gemma:2b in limited GPU VRAM
  - LoRA (Low-Rank Adaptation) to train only ~1-2% of parameters
  - SFTTrainer from TRL for supervised fine-tuning

Unlike the Ollama Modelfile approach (prompt engineering only),
this actually updates model weights to learn the classification task.

Requirements:
  - CUDA GPU with 8GB+ VRAM
  - peft, trl, bitsandbytes, transformers, accelerate, torch

Usage:
    python -m training.finetune_qlora

Output:
  - Adapter weights saved to training/saved_model/qlora_adapter/
  - Can be merged with base model or loaded alongside it
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from app.config import SUPPORTED_CATEGORIES, config

logger = logging.getLogger(__name__)

# Training hyperparameters
MAX_SEQ_LENGTH: int = 512
LORA_RANK: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
TARGET_MODULES: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Classification prompt template for training
CLASSIFICATION_TEMPLATE: str = (
    "Classify the following sales representative field note into exactly "
    "one of these categories: {categories}\n\n"
    "Sales Note: {note}\n\n"
    "Category: {category}"
)

SYSTEM_PROMPT: str = (
    "You are a sales intelligence analyst for an FMCG company. "
    "Your task is to classify sales representative field notes into "
    "exactly one of these 5 issue categories:\n"
    "1. supply_chain_delay — stock shortages, delivery delays, replenishment issues\n"
    "2. retailer_dissatisfaction — complaints, unhappy retailers, service issues\n"
    "3. pricing_conflict — price disputes, margin concerns, discount conflicts\n"
    "4. competitor_pressure — competitor actions, market share threats, rival offers\n"
    "5. demand_spike — unexpected demand surges, stockouts from high volume\n\n"
    "Respond with ONLY the category name. No explanation."
)


def _format_categories() -> str:
    """Format category list for prompt."""
    return ", ".join(SUPPORTED_CATEGORIES)


def _row_to_text(row: Dict[str, str]) -> str:
    """Convert a dataset row to training text format.

    Args:
        row: Dict with 'input' (note) and 'output' (category) keys.

    Returns:
        Formatted training text.
    """
    return (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{row['input']}\n\n"
        f"### Response:\n{row['output']}"
    )


def _load_dataset_for_qlora() -> Dict[str, List[Dict[str, str]]]:
    """Load and split the dataset for QLoRA training.

    Returns:
        Dict with 'train' and 'test' keys containing lists of
        {'input': ..., 'output': ...} dicts.
    """
    from sklearn.model_selection import train_test_split

    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df = pd.read_excel(config.dataset_path)
    df = df[["rep_note", "issue_category"]].dropna()
    df["input"] = df["rep_note"].str.strip()
    df["output"] = df["issue_category"].map(config.category_map if hasattr(config, "category_map") else {
        "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
        "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
        "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
        "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
        "DEMAND_SURGE": "demand_spike",
    })
    df = df.dropna(subset=["output"])

    # Deduplicate to prevent data leakage
    before = len(df)
    df = df.drop_duplicates(subset=["input"])
    after = len(df)
    if before > after:
        logger.info("Removed %d duplicate notes", before - after)

    train_df, test_df = train_test_split(
        df, test_size=config.test_size, random_state=config.random_seed,
        stratify=df["output"],
    )

    return {
        "train": train_df[["input", "output"]].to_dict("records"),
        "test": test_df[["input", "output"]].to_dict("records"),
    }


def _create_hf_dataset(data: List[Dict[str, str]]) -> Any:
    """Create a HuggingFace Dataset from the data.

    Args:
        data: List of {'input': ..., 'output': ...} dicts.

    Returns:
        HuggingFace Dataset with 'text' column.
    """
    from datasets import Dataset

    texts = [_row_to_text(row) for row in data]
    return Dataset.from_dict({"text": texts})


def finetune(
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run QLoRA fine-tuning on gemma:2b.

    Args:
        output_dir: Directory to save adapter weights.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        batch_size: Training batch size.

    Returns:
        Dictionary with training results and metrics.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    output_dir = output_dir or config.qlora_adapter_path
    epochs = epochs or config.qlora_epochs
    learning_rate = learning_rate or config.qlora_learning_rate
    batch_size = batch_size or config.qlora_batch_size

    os.makedirs(output_dir, exist_ok=True)

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for QLoRA training. "
            "Use CPU training with the sklearn pipeline instead."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)

    if gpu_mem < 6:
        logger.warning(
            "GPU has only %.1f GB VRAM. 8GB+ recommended for QLoRA. "
            "Consider reducing batch_size or using gradient accumulation.",
            gpu_mem,
        )

    # Load dataset
    logger.info("Loading dataset from %s", config.dataset_path)
    data = _load_dataset_for_qlora()
    train_dataset = _create_hf_dataset(data["train"])
    eval_dataset = _create_hf_dataset(data["test"])
    logger.info("Train: %d samples, Eval: %d samples", len(train_dataset), len(eval_dataset))

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model_name = config.model_name if hasattr(config, "model_name") and "/" in config.model_name else "google/gemma-2b"
    logger.info("Loading base model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=config.random_seed,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # Train
    logger.info("Starting QLoRA fine-tuning...")
    train_result = trainer.train()

    # Save adapter
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        "method": "QLoRA",
        "base_model": model_name,
        "train_samples": len(data["train"]),
        "eval_samples": len(data["test"]),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": train_result.training_loss,
        "gpu": gpu_name,
    }
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("QLoRA fine-tuning complete. Adapter saved to %s", output_dir)
    logger.info("Training loss: %.4f", train_result.training_loss)

    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        results = finetune()
        print(f"\n✅ QLoRA fine-tuning complete!")
        print(f"   Adapter saved to: {config.qlora_adapter_path}")
        print(f"   Training loss: {results.get('train_loss', 'N/A'):.4f}")
        print(f"   Trainable params: {results['trainable_params']:,} / {results['total_params']:,}")
    except RuntimeError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
