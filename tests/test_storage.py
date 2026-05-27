"""Comprehensive tests for the prediction storage service.

Tests saving, retrieval, category counting, and error handling.
"""

import json
from typing import Dict, List

import pytest

from app.services.storage import CSV_COLUMNS, PredictionStorage


class TestStorageInitialization:
    """Tests for storage initialization."""

    def test_creates_directories(self, tmp_path) -> None:
        import os

        nested_path = str(tmp_path / "nested" / "deep" / "predictions.csv")
        jsonl_path = str(tmp_path / "nested" / "deep" / "predictions.jsonl")

        storage = PredictionStorage(csv_path=nested_path, jsonl_path=jsonl_path)
        assert os.path.exists(os.path.dirname(nested_path))

    def test_creates_csv_header(self, temp_storage: PredictionStorage) -> None:
        import os

        assert os.path.exists(temp_storage.csv_path)
        with open(temp_storage.csv_path, "r") as f:
            header = f.readline().strip()
            assert header == ",".join(CSV_COLUMNS)


class TestSavePrediction:
    """Tests for saving predictions."""

    def test_save_writes_to_csv(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("test note", "supply_chain_delay")

        with open(temp_storage.csv_path, "r") as f:
            lines = f.readlines()
            # Header + 1 record
            assert len(lines) == 2

    def test_save_writes_to_jsonl(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("test note", "supply_chain_delay")

        with open(temp_storage.jsonl_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1

    def test_save_returns_record(self, temp_storage: PredictionStorage) -> None:
        record = temp_storage.save_prediction("test note", "supply_chain_delay")
        assert record["input_note"] == "test note"
        assert record["issue_category"] == "supply_chain_delay"
        assert "timestamp" in record

    def test_save_multiple(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("note 1", "supply_chain_delay")
        temp_storage.save_prediction("note 2", "pricing_conflict")
        temp_storage.save_prediction("note 3", "demand_spike")

        history = temp_storage.get_prediction_history()
        assert len(history) == 3

    def test_jsonl_content_is_valid_json(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("test note", "supply_chain_delay")

        with open(temp_storage.jsonl_path, "r") as f:
            record = json.loads(f.readline())
            assert isinstance(record, dict)
            assert record["input_note"] == "test note"


class TestPredictionHistory:
    """Tests for retrieving prediction history."""

    def test_empty_history(self, temp_storage: PredictionStorage) -> None:
        history = temp_storage.get_prediction_history()
        assert history == []

    def test_reverse_chronological_order(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("first", "supply_chain_delay")
        temp_storage.save_prediction("second", "pricing_conflict")

        history = temp_storage.get_prediction_history()
        assert history[0]["input_note"] == "second"
        assert history[1]["input_note"] == "first"

    def test_limit_parameter(self, temp_storage: PredictionStorage) -> None:
        for i in range(10):
            temp_storage.save_prediction(f"note {i}", "supply_chain_delay")

        history = temp_storage.get_prediction_history(limit=3)
        assert len(history) == 3

    def test_nonexistent_jsonl(self, tmp_path) -> None:
        storage = PredictionStorage(
            csv_path=str(tmp_path / "a.csv"),
            jsonl_path=str(tmp_path / "nonexistent.jsonl"),
        )
        storage._ensure_csv_header()

        history = storage.get_prediction_history()
        assert history == []

    def test_limit_greater_than_count(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("note", "supply_chain_delay")
        history = temp_storage.get_prediction_history(limit=100)
        assert len(history) == 1


class TestCategoryCounts:
    """Tests for category distribution counting."""

    def test_empty_counts(self, temp_storage: PredictionStorage) -> None:
        counts = temp_storage.get_category_counts()
        assert counts == {}

    def test_single_category(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("note 1", "supply_chain_delay")
        temp_storage.save_prediction("note 2", "supply_chain_delay")

        counts = temp_storage.get_category_counts()
        assert counts["supply_chain_delay"] == 2

    def test_multiple_categories(self, temp_storage: PredictionStorage) -> None:
        temp_storage.save_prediction("n1", "supply_chain_delay")
        temp_storage.save_prediction("n2", "supply_chain_delay")
        temp_storage.save_prediction("n3", "pricing_conflict")
        temp_storage.save_prediction("n4", "demand_spike")
        temp_storage.save_prediction("n5", "demand_spike")
        temp_storage.save_prediction("n6", "demand_spike")

        counts = temp_storage.get_category_counts()
        assert counts.get("supply_chain_delay") == 2
        assert counts.get("pricing_conflict") == 1
        assert counts.get("demand_spike") == 3

    def test_unknown_category(self, tmp_path) -> None:
        storage = PredictionStorage(
            csv_path=str(tmp_path / "a.csv"),
            jsonl_path=str(tmp_path / "a.jsonl"),
        )
        storage._ensure_csv_header()
        storage._ensure_directories()

        # Manually write a record with "unknown" category
        record = {
            "timestamp": "2024-01-01T00:00:00",
            "input_note": "test",
            "issue_category": "unknown",
        }
        import json as json_mod
        with open(storage.jsonl_path, "w") as f:
            f.write(json_mod.dumps(record) + "\n")

        counts = storage.get_category_counts()
        assert counts.get("unknown") == 1
