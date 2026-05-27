"""Storage service for prediction logging and history management."""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CSV_PATH: str = "data/predictions.csv"
JSONL_PATH: str = "data/predictions.jsonl"


class PredictionStorage:
    """Handles storage of predictions in CSV and JSONL formats."""

    def __init__(
        self,
        csv_path: str = CSV_PATH,
        jsonl_path: str = JSONL_PATH,
    ) -> None:
        """Initialize storage with configured paths.

        Args:
            csv_path: Path to CSV prediction log.
            jsonl_path: Path to JSONL prediction log.
        """
        self.csv_path: str = csv_path
        self.jsonl_path: str = jsonl_path
        self._ensure_directories()
        self._ensure_csv_header()
        logger.info(
            "PredictionStorage initialized: csv=%s, jsonl=%s",
            csv_path,
            jsonl_path,
        )

    def _ensure_directories(self) -> None:
        """Create directories if they do not exist."""
        for path in [self.csv_path, self.jsonl_path]:
            directory: str = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.debug("Created directory: %s", directory)

    def _ensure_csv_header(self) -> None:
        """Write CSV header if file does not exist or is empty."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "input_note",
                    "issue_category",
                ])
            logger.debug("Created CSV with header: %s", self.csv_path)

    def save_prediction(
        self,
        input_note: str,
        issue_category: str,
    ) -> Dict[str, Any]:
        """Save prediction to both CSV and JSONL.

        Args:
            input_note: The original input note.
            issue_category: The predicted category.

        Returns:
            Dictionary with saved prediction data.
        """
        timestamp: str = datetime.now().isoformat()
        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "input_note": input_note,
            "issue_category": issue_category,
        }

        self._save_to_csv(record)
        self._save_to_jsonl(record)

        logger.info("Prediction saved: category=%s", issue_category)
        return record

    def _save_to_csv(self, record: Dict[str, Any]) -> None:
        """Append record to CSV file."""
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["timestamp"],
                    record["input_note"],
                    record["issue_category"],
                ])
        except IOError as e:
            logger.error("Failed to write CSV: %s", e)
            raise

    def _save_to_jsonl(self, record: Dict[str, Any]) -> None:
        """Append record to JSONL file."""
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except IOError as e:
            logger.error("Failed to write JSONL: %s", e)
            raise

    def get_prediction_history(
        self, limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve prediction history from JSONL.

        Args:
            limit: Maximum number of records to return (most recent first).

        Returns:
            List of prediction records.
        """
        history: List[Dict[str, Any]] = []
        if not os.path.exists(self.jsonl_path):
            return history

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Failed to read JSONL: %s", e)
            raise

        history.reverse()
        if limit:
            history = history[:limit]
        return history

    def get_category_counts(self) -> Dict[str, int]:
        """Get count of predictions per category.

        Returns:
            Dictionary mapping category names to counts.
        """
        counts: Dict[str, int] = {}
        history: List[Dict[str, Any]] = self.get_prediction_history()
        for record in history:
            category: str = record.get("issue_category", "unknown")
            counts[category] = counts.get(category, 0) + 1
        return counts
