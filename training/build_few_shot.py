"""Build optimized few-shot prompts from the training data.

Selects the most representative examples per category using
stratified sampling to maximize classification accuracy when
embedded in the Ollama system prompt.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from app.config import SUPPORTED_CATEGORIES, config

logger = logging.getLogger(__name__)


def build_few_shot_examples(
    df: pd.DataFrame,
    n_per_category: int = 6,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Build few-shot examples optimized for prompt embedding.

    Selects diverse examples covering different note lengths and
    phrasings within each category.

    Args:
        df: DataFrame with 'input' and 'output' columns.
        n_per_category: Number of examples per category.
        seed: Random seed.

    Returns:
        Dict mapping category to list of {"input": ..., "output": ...} dicts.
    """
    import random
    random.seed(seed)

    examples: Dict[str, List[Dict[str, str]]] = {}

    for category in SUPPORTED_CATEGORIES:
        cat_df = df[df["output"] == category]
        notes = cat_df["input"].tolist()

        if len(notes) <= n_per_category:
            selected = notes
        else:
            # Stratified selection by note length for diversity
            notes_with_len = sorted(notes, key=len)
            selected = []
            bucket_size = len(notes_with_len) // n_per_category

            for i in range(n_per_category):
                start = i * bucket_size
                end = min(start + bucket_size, len(notes_with_len))
                bucket = notes_with_len[start:end]
                if bucket:
                    selected.append(random.choice(bucket))

            # Fill remaining slots randomly
            while len(selected) < n_per_category:
                candidate = random.choice(notes)
                if candidate not in selected:
                    selected.append(candidate)

        examples[category] = [
            {"input": note, "output": category} for note in selected
        ]

        logger.info(
            "Selected %d examples for category '%s'",
            len(selected), category,
        )

    return examples


def export_training_jsonl(
    df: pd.DataFrame,
    output_path: str = "training/dataset/ollama_training.jsonl",
) -> str:
    """Export the full dataset in JSONL format for reference.

    This file can be used for manual inspection or external tools.

    Args:
        df: DataFrame with 'input' and 'output' columns.
        output_path: Output file path.

    Returns:
        Path to the written file.
    """
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "input": row["input"],
                "output": row["output"],
            }
            f.write(json.dumps(record) + "\n")

    logger.info("Exported %d training records to %s", len(df), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from training.prepare_dataset import prepare_dataset

    train_df, test_df = prepare_dataset()
    examples = build_few_shot_examples(train_df)

    print("\nFew-shot examples built:")
    for cat, exs in examples.items():
        print(f"  {cat}: {len(exs)} examples")

    export_training_jsonl(train_df)
    print("\nTraining data exported to training/dataset/ollama_training.jsonl")
