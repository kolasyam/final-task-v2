"""Generate an Ollama Modelfile for prompt engineering gemma:2b.

This module converts the 500-record sales dataset into an Ollama Modelfile
that creates a prompt-engineered model 'gemma-sales-intel' based on gemma:2b.

IMPORTANT: This is NOT parameter fine-tuning — model weights are NOT updated.
The Modelfile embeds few-shot examples and category definitions in the system
prompt to guide the base model's responses.

For real parameter fine-tuning, use training/finetune_qlora.py (QLoRA) instead.

The Modelfile uses:
  - System prompt with category definitions + few-shot examples
  - Inference parameters optimized for classification

Usage:
    python -m training.generate_modelfile
    Then: ollama create gemma-sales-intel -f training/Modelfile

This creates a new Ollama model specialized for sales note classification
that the system will automatically detect and use.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.config import (
    CATEGORY_MAP,
    SUPPORTED_CATEGORIES,
    config,
)

logger = logging.getLogger(__name__)

# Output path for the generated Modelfile
MODELFILE_PATH: str = os.path.join(config.model_dir, "Modelfile")
MODELFILE_BACKUP_PATH: str = os.path.join(config.model_dir, "Modelfile.bak")

# Ollama model name for the fine-tuned model
PROMPT_MODEL_NAME: str = "gemma-sales-intel"

# Number of few-shot examples per category to embed in system prompt
EXAMPLES_PER_CATEGORY: int = 6

# Number of training TEMPLATE blocks for fine-tuning (higher = more training)
TRAINING_PAIRS_PER_CATEGORY: int = 12


def _load_dataset() -> pd.DataFrame:
    """Load and preprocess the dataset.

    Returns:
        DataFrame with 'input' (note text) and 'output' (category label).
    """
    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df = pd.read_excel(config.dataset_path)
    df = df[["rep_note", "issue_category"]].dropna()
    df["input"] = df["rep_note"].str.strip()
    df["output"] = df["issue_category"].map(CATEGORY_MAP)
    df = df.dropna(subset=["output"])

    logger.info("Loaded %d records from %s", len(df), config.dataset_path)
    return df


def _select_few_shot_examples(
    df: pd.DataFrame,
    n_per_category: int = EXAMPLES_PER_CATEGORY,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """Selects representative few-shot examples for each category.

    Picks diverse examples to cover different phrasings within each category.

    Args:
        df: Dataset DataFrame.
        n_per_category: Number of examples per category.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping category name to list of (note, category) tuples.
    """
    random.seed(seed)
    examples: Dict[str, List[Tuple[str, str]]] = {}

    for category in SUPPORTED_CATEGORIES:
        cat_df = df[df["output"] == category]
        # Select diverse samples: short, medium, and long notes
        notes = cat_df["input"].tolist()

        if len(notes) <= n_per_category:
            selected = notes
        else:
            # Sort by length and pick from different positions for diversity
            notes_sorted = sorted(notes, key=len)
            step = max(1, len(notes_sorted) // n_per_category)
            selected = [
                notes_sorted[i * step]
                for i in range(min(n_per_category, len(notes_sorted) // step))
            ]
            # Fill remaining with random if needed
            while len(selected) < n_per_category and len(notes_sorted) > len(selected):
                candidate = random.choice(notes_sorted)
                if candidate not in selected:
                    selected.append(candidate)

        examples[category] = [(note, category) for note in selected]

    return examples


def _build_system_prompt(
    few_shot_examples: Dict[str, List[Tuple[str, str]]],
) -> str:
    """Build the system prompt with category definitions and few-shot examples.

    This prompt teaches the model the classification task by providing:
    - Clear category definitions with keywords illustrative patterns
    - Multiple labeled examples per category

    Args:
        few_shot_examples: Dict of category -> [(note, label), ...].

    Returns:
        Complete system prompt string.
    """
    lines: List[str] = [
        "You are a sales intelligence expert for a FMCG/consumer goods company. "
        "Your ONLY task is to classify sales representative field notes into "
        "exactly one of these 5 issue categories:\n",
    ]

    # Category definitions
    definitions = {
        "supply_chain_delay": (
            "Stock shortages, delivery delays, replenishment issues, "
            "inventory mismatch, products running out, slow stock movement, "
            "warehouse follow-up, backorders"
        ),
        "retailer_dissatisfaction": (
            "Retailer complaints, unhappiness with service, relationship "
            "issues, frustration, anger, poor experience, service quality "
            "problems, threats to reduce business"
        ),
        "pricing_conflict": (
            "Price disputes, margin concerns, pricing disagreements, "
            "discount conflicts, billing issues, rates too high, "
            "pricing policy complaints"
        ),
        "competitor_pressure": (
            "Competitor actions, rival campaigns, retailers switching to "
            "competitors, competitor pricing, market share threats, "
            "competitive product launches"
        ),
        "demand_spike": (
            "Unexpected demand surges, demand overflow, stockouts from "
            "high volume, sudden rush, orders exceeding forecast, "
            "supply couldn't meet demand"
        ),
    }

    for cat in SUPPORTED_CATEGORIES:
        display_name = cat.replace("_", " ").title()
        definition = definitions.get(cat, "")
        lines.append(f"{cat} — {display_name}: {definition}")

    # Few-shot examples
    lines.append("\nHere are examples of how to classify notes:\n")
    lines.append("---")

    for category in SUPPORTED_CATEGORIES:
        if category in few_shot_examples:
            for note, label in few_shot_examples[category]:
                lines.append(f'Input: "{note}"')
                lines.append(f"Output: {label}")
                lines.append("---")

    lines.append(
        "\nRULES:\n"
        "- Respond with ONLY the exact category name (e.g., supply_chain_delay)\n"
        "- Do NOT add explanations, reasoning, or extra text\n"
        "- Choose the SINGLE best category\n"
        "- If unsure, choose the category closest to the note's main issue"
    )

    return "\n".join(lines)


def generate_modelfile(
    output_path: str = MODELFILE_PATH,
    examples_per_category: int = EXAMPLES_PER_CATEGORY,
) -> str:
    """Generate the complete Ollama Modelfile.

    Creates a Modelfile that:
    1. Uses gemma:2b as the base model
    2. Sets a specialized system prompt with few-shot examples
    3. Configures inference parameters for classification

    Args:
        output_path: Where to write the Modelfile.
        examples_per_category: Number of few-shot examples per category.

    Returns:
        The generated Modelfile content as a string.
    """
    logger.info("Generating Modelfile for Ollama fine-tuning...")

    # Load data and select examples
    df = _load_dataset()
    few_shot = _select_few_shot_examples(df, examples_per_category)
    system_prompt = _build_system_prompt(few_shot)

    # Count total examples
    total_examples = sum(len(v) for v in few_shot.values())
    logger.info(
        "Selected %d few-shot examples across %d categories",
        total_examples, len(few_shot),
    )

    # Build the Modelfile
    lines: List[str] = [
        "# ============================================",
        "# Sales Intelligence Classifier Modelfile",
        "# Based on gemma:2b — fine-tuned for classifying",
        "# sales representative field notes",
        "# ============================================",
        "# Auto-generated by training.generate_modelfile",
        "",
        f"FROM {config.model_name}",
        "",
        "# ---- Inference Parameters ----",
        "PARAMETER temperature 0.1",
        "PARAMETER top_p 0.9",
        "PARAMETER top_k 40",
        "PARAMETER num_predict 30",
        "PARAMETER repeat_penalty 1.1",
        "",
        "# ---- System Prompt ----",
        "SYSTEM \"\"\"",
        system_prompt,
        "\"\"\"",
        "",
        "# ---- Template ----",
        'TEMPLATE """{{ if .System }}<start_of_turn>model',
        "{{ .System }}<end_of_turn>",
        "{{ end }}{{ if .Prompt }}<start_of_turn>user",
        "{{ .Prompt }}<end_of_turn>",
        '{{ end }}<start_of_turn>model',
        "{{ .Response }}<end_of_turn>\"\"\"",
        "",
    ]

    modelfile_content = "\n".join(lines)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Backup existing Modelfile
    if os.path.exists(output_path):
        import shutil
        shutil.copy2(output_path, MODELFILE_BACKUP_PATH)
        logger.info("Backed up existing Modelfile to %s", MODELFILE_BACKUP_PATH)

    # Write Modelfile
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    logger.info("Modelfile generated at: %s", output_path)
    logger.info(
        "To create the fine-tuned model, run:\n"
        "  ollama create %s -f %s",
        PROMPT_MODEL_NAME,
        output_path,
    )

    return modelfile_content


def create_ollama_model(
    modelfile_path: str = MODELFILE_PATH,
    model_name: str = PROMPT_MODEL_NAME,
) -> bool:
    """Create the fine-tuned Ollama model using the CLI.

    Requires Ollama to be running locally.

    Args:
        modelfile_path: Path to the generated Modelfile.
        model_name: Name for the new Ollama model.

    Returns:
        True if the model was created successfully.
    """
    import subprocess

    if not os.path.exists(modelfile_path):
        logger.error("Modelfile not found at %s. Run generate_modelfile() first.")
        return False

    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Successfully created Ollama model: %s", model_name)
            return True
        else:
            logger.error("Ollama create failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error(
            "Ollama CLI not found. Ensure Ollama is installed and in PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama create timed out after 600 seconds.")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Step 1: Generate the Modelfile
    modelfile = generate_modelfile()

    # Step 2: Attempt to create the Ollama model
    logger.info("\n" + "=" * 60)
    logger.info("Creating fine-tuned Ollama model...")
    logger.info("=" * 60)

    success = create_ollama_model()

    if success:
        print(f"\n✅ Fine-tuned model '{PROMPT_MODEL_NAME}' created successfully!")
        print(f"   The system will automatically use it for predictions.")
    else:
        print(f"\n⚠️  Could not auto-create the model. Run manually:")
        print(f"   ollama create {PROMPT_MODEL_NAME} -f {MODELFILE_PATH}")
