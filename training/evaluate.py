"""Evaluation pipeline for the scikit-learn classifier.

Loads persisted model artifacts, generates predictions on the test set,
and computes comprehensive metrics using scikit-learn.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.config import config

logger = logging.getLogger(__name__)


def _load_artifacts() -> tuple:
    """Load the persisted model artifacts.

    Returns:
        Tuple of (vectorizer, classifier, label_encoder).

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    vectorizer = joblib.load(config.vectorizer_path)
    classifier = joblib.load(config.classifier_path)
    label_encoder = joblib.load(config.label_encoder_path)
    logger.info("All artifacts loaded successfully")
    return vectorizer, classifier, label_encoder


def evaluate(
    test_df,
    output_dir: str = config.model_dir,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.

    Loads the trained model, generates predictions on the test set,
    and computes all evaluation metrics.

    Args:
        test_df: Test DataFrame with 'input' and 'output' columns.
        output_dir: Directory to save evaluation artifacts.

    Returns:
        Dictionary of evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    vectorizer, classifier, label_encoder = _load_artifacts()

    # Prepare features and labels
    X_test = vectorizer.transform(test_df["input"].values)
    y_true = label_encoder.transform(test_df["output"].values)

    # Generate predictions
    logger.info("Generating predictions for %d test samples", len(y_true))
    y_pred = classifier.predict(X_test)

    # Compute metrics
    metrics = _compute_metrics(y_true, y_pred, label_encoder, output_dir)

    # Save full classification report
    target_names = label_encoder.classes_.tolist()
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0,
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Classification report saved to %s", report_path)

    # Print summary
    _print_summary(metrics)

    return metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any,
    output_dir: str,
) -> Dict[str, Any]:
    """Compute and save evaluation metrics.

    Args:
        y_true: True label indices.
        y_pred: Predicted label indices.
        label_encoder: Fitted label encoder.
        output_dir: Directory to save metrics file.

    Returns:
        Dictionary of evaluation metrics.
    """
    target_names = label_encoder.classes_.tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True,
        zero_division=0,
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    # Save serializable metrics
    serializable = {
        k: v for k, v in metrics.items() if k != "confusion_matrix"
    }
    serializable["confusion_matrix"] = conf_matrix.tolist()
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics


def _print_summary(metrics: Dict[str, Any]) -> None:
    """Print formatted evaluation summary.

    Args:
        metrics: Evaluation metrics dictionary.
    """
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info("Accuracy:          %.4f", metrics["accuracy"])
    logger.info("Precision (avg):   %.4f", metrics["precision_weighted"])
    logger.info("Recall (avg):      %.4f", metrics["recall_weighted"])
    logger.info("F1 Score (avg):    %.4f", metrics["f1_weighted"])
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from training.prepare_dataset import prepare_dataset

    train_df, test_df = prepare_dataset()
    metrics = evaluate(test_df)

    print(f"\nEvaluation complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_weighted']:.4f}")
