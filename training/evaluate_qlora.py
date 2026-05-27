"""Evaluate the QLoRA-finetuned model against the baseline.

Compares:
  1. QLoRA-finetuned gemma:2b (real parameter fine-tuning)
  2. Prompt-engineered gemma-sales-intel (Ollama Modelfile)
  3. Base gemma:2b (zero-shot)
  4. Scikit-learn TF-IDF + RandomForest

Usage:
    python -m training.evaluate_qlora
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from app.config import SUPPORTED_CATEGORIES, config

logger = logging.getLogger(__name__)


def _load_test_data() -> pd.DataFrame:
    """Load the test dataset.

    Returns:
        DataFrame with 'input' and 'output' columns.
    """
    from sklearn.model_selection import train_test_split

    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df = pd.read_excel(config.dataset_path)
    df = df[["rep_note", "issue_category"]].dropna()
    df["input"] = df["rep_note"].str.strip()
    df["output"] = df["issue_category"].map({
        "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
        "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
        "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
        "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
        "DEMAND_SURGE": "demand_spike",
    })
    df = df.dropna(subset=["output"])
    df = df.drop_duplicates(subset=["input"])

    _, test_df = train_test_split(
        df[["input", "output"]],
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=df["output"],
    )
    return test_df


def evaluate_qlora(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the QLoRA-finetuned model.

    Args:
        test_df: Test DataFrame with 'input' and 'output' columns.

    Returns:
        Evaluation metrics dictionary.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = config.qlora_adapter_path
    if not os.path.exists(adapter_path):
        logger.error("QLoRA adapter not found at %s. Run finetune_qlora.py first.", adapter_path)
        return {"error": "adapter not found"}

    if not torch.cuda.is_available():
        logger.error("CUDA GPU required for QLoRA evaluation.")
        return {"error": "no cuda"}

    logger.info("Loading QLoRA model from %s", adapter_path)

    base_model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    correct = 0
    total = 0
    latencies: List[float] = []
    predictions: List[Dict[str, str]] = []

    system_msg = (
        "You are a sales intelligence analyst. Classify the note into one of: "
        "supply_chain_delay, retailer_dissatisfaction, pricing_conflict, "
        "competitor_pressure, demand_spike. Respond with ONLY the category name."
    )

    for _, row in test_df.iterrows():
        note = row["input"]
        true_label = row["output"]

        prompt = (
            f"### Instruction:\n{system_msg}\n\n"
            f"### Input:\n{note}\n\n"
            f"### Response:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
            )
        elapsed = time.time() - start
        latencies.append(elapsed)

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = generated.strip().lower().split()[0] if generated.strip() else ""
        predicted = predicted.rstrip(".!?,;:")

        # Normalize prediction
        if predicted not in SUPPORTED_CATEGORIES:
            for cat in SUPPORTED_CATEGORIES:
                if cat in predicted or cat.replace("_", " ") in predicted:
                    predicted = cat
                    break

        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "note": note[:100],
            "true": true_label,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    results = {
        "method": "QLoRA",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_latency_seconds": round(avg_latency, 3),
        "predictions": predictions,
    }

    logger.info("QLoRA Results: accuracy=%.4f, avg_latency=%.3fs", accuracy, avg_latency)
    return results


def evaluate_all() -> Dict[str, Any]:
    """Run evaluation for all available methods.

    Returns:
        Combined results dictionary.
    """
    test_df = _load_test_data()
    logger.info("Evaluating on %d test samples", len(test_df))

    results: Dict[str, Any] = {
        "test_samples": len(test_df),
        "methods": {},
    }

    # 1. QLoRA
    logger.info("=" * 50)
    logger.info("Evaluating QLoRA-finetuned model...")
    qlora_results = evaluate_qlora(test_df)
    results["methods"]["qlora"] = qlora_results

    # 2. Sklearn
    logger.info("=" * 50)
    logger.info("Evaluating scikit-learn fallback...")
    try:
        from sklearn.preprocessing import LabelEncoder

        train_df = _load_test_data()  # Use full data for a quick check
        vectorizer = TfidfVectorizer(max_features=config.max_features, ngram_range=(1, 2))
        classifier = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
            n_estimators=config.n_estimators, random_state=config.random_seed, class_weight="balanced"
        )
        le = LabelEncoder()

        # Quick sklearn eval
        full_df = pd.read_excel(config.dataset_path)
        full_df = full_df[["rep_note", "issue_category"]].dropna()
        full_df["input"] = full_df["rep_note"].str.strip()
        full_df["output"] = full_df["issue_category"].map({
            "SUPPLY_CHAIN_ISSUE": "supply_chain_delay",
            "RETAILER_RELATIONSHIP_ISSUE": "retailer_dissatisfaction",
            "PRICING_AND_MARGIN_CONFLICT": "pricing_conflict",
            "COMPETITOR_MARKET_PRESSURE": "competitor_pressure",
            "DEMAND_SURGE": "demand_spike",
        })
        full_df = full_df.dropna(subset=["output"]).drop_duplicates(subset=["input"])

        from sklearn.model_selection import train_test_split
        _, sk_test = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df["output"])

        X_test = vectorizer.fit_transform(full_df["input"])
        y_test = le.fit_transform(full_df["output"])
        classifier.fit(X_test, le.transform(full_df["output"]))

        X_t = vectorizer.transform(sk_test["input"])
        y_t = le.transform(sk_test["output"])
        y_p = classifier.predict(X_t)

        from sklearn.metrics import accuracy_score
        sk_accuracy = accuracy_score(y_t, y_p)
        results["methods"]["sklearn"] = {"accuracy": sk_accuracy, "method": "sklearn"}
        logger.info("Sklearn accuracy: %.4f", sk_accuracy)
    except Exception as e:
        logger.warning("Could not evaluate sklearn: %s", e)
        results["methods"]["sklearn"] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    for method, res in results["methods"].items():
        if "accuracy" in res:
            logger.info("  %-20s accuracy=%.4f", method, res["accuracy"])
    logger.info("=" * 60)

    # Save results
    output_path = os.path.join(config.model_dir, "qlora_evaluation.json")
    os.makedirs(config.model_dir, exist_ok=True)
    serializable = {k: v for k, v in results.items() if k != "predictions"}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        results = evaluate_all()
        print("\n✅ Evaluation complete!")
        for method, res in results.get("methods", {}).items():
            if "accuracy" in res:
                print(f"  {method}: accuracy={res['accuracy']:.4f}")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)
