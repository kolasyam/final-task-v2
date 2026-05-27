"""Scikit-learn training pipeline for sales note classification.

Trains a TF-IDF vectorizer + RandomForest classifier on the
standardized sales dataset. Persists artifacts for inference.
Tracks all experiments with MLflow for reproducibility.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import config

logger = logging.getLogger(__name__)


def _build_pipeline() -> Tuple[Pipeline, LabelEncoder]:
    """Build the scikit-learn classification pipeline.

    Returns:
        Tuple of (sklearn Pipeline, fitted LabelEncoder).
    """
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config.random_seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    label_encoder = LabelEncoder()

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    return pipeline, label_encoder


def train(
    train_df,
    test_df,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate the sales note classifier.

    Steps:
        1. Encode labels with LabelEncoder.
        2. Train TF-IDF + RandomForest pipeline.
        3. Evaluate on test set (accuracy, precision, recall, F1).
        4. Persist all artifacts (vectorizer, classifier, label encoder).
        5. Log parameters and metrics to MLflow.

    Args:
        train_df: Training DataFrame with 'input' and 'output' columns.
        test_df: Test DataFrame with 'input' and 'output' columns.
        output_dir: Directory to save model artifacts. Defaults to config.

    Returns:
        Dictionary of evaluation metrics.
    """
    if output_dir is None:
        output_dir = config.model_dir

    os.makedirs(output_dir, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name="sklearn-tfidf-rf"):
        # Log configuration
        mlflow.log_params({
            "model_type": "TF-IDF + RandomForest",
            "vectorizer": "TfidfVectorizer",
            "max_features": config.max_features,
            "ngram_range": "(1, 2)",
            "classifier": "RandomForestClassifier",
            "n_estimators": config.n_estimators,
            "class_weight": "balanced",
            "test_size": config.test_size,
            "random_seed": config.random_seed,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        })

        # Build pipeline
        pipeline, label_encoder = _build_pipeline()

        # Encode labels
        y_train = label_encoder.fit_transform(train_df["output"])
        y_test = label_encoder.transform(test_df["output"])
        X_train = train_df["input"].values
        X_test = test_df["input"].values

        # Cross-validation on training set (detects overfitting)
        logger.info("Running 5-fold cross-validation on training set...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")
        logger.info("CV F1 scores: %s", cv_scores)
        logger.info("CV F1 mean: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())

        if cv_scores.mean() > 0.99:
            logger.warning(
                "⚠ CV F1 is %.4f — possible overfitting or data leakage. "
                "Ensure train/test splits have no duplicate records.",
                cv_scores.mean(),
            )

        # Train on full training set
        logger.info(
            "Training TF-IDF + RandomForest on %d samples", len(X_train),
        )
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        logger.info("Training completed in %.2f seconds", training_time)

        # Evaluate on held-out test set
        y_pred = pipeline.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred, label_encoder)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        })

        # Persist artifacts
        _save_artifacts(pipeline, label_encoder, output_dir)

        # Log artifacts to MLflow
        mlflow.log_artifacts(output_dir, artifact_path="model")

        # Print summary
        _print_summary(metrics)

        return metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: True label indices.
        y_pred: Predicted label indices.
        label_encoder: Fitted label encoder for class names.

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

    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
    }


def _save_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    output_dir: str,
) -> None:
    """Persist model artifacts to disk.

    Saves the vectorizer, classifier, and label encoder separately
    for flexible loading during inference.

    Args:
        pipeline: Fitted sklearn Pipeline.
        label_encoder: Fitted LabelEncoder.
        output_dir: Directory to save artifacts.
    """
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    classifier_path = os.path.join(output_dir, "classifier.joblib")
    label_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    metrics_path = os.path.join(output_dir, "training_metadata.json")

    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    joblib.dump(pipeline.named_steps["clf"], classifier_path)
    joblib.dump(label_encoder, label_encoder_path)

    metadata = {
        "vectorizer_path": vectorizer_path,
        "classifier_path": classifier_path,
        "label_encoder_path": label_encoder_path,
        "classes": label_encoder.classes_.tolist(),
        "pipeline_steps": [name for name, _ in pipeline.steps],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model artifacts saved to %s", output_dir)


def _print_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted training summary.

    Args:
        metrics: Evaluation metrics dictionary.
    """
    logger.info("=" * 50)
    logger.info("TRAINING RESULTS")
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
    metrics = train(train_df, test_df)

    print(f"\nTraining complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_weighted']:.4f}")
