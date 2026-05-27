"""Dataset preparation: reads the standardized Excel sales dataset.

Reads from the configured .xlsx file (500 records, 5 categories),
extracts rep_note and issue_category, maps categories to normalized
names, and splits 80/20 into train/test sets for scikit-learn training.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import CATEGORY_MAP, config

logger = logging.getLogger(__name__)


def _load_source_data(path: str) -> pd.DataFrame:
    """Load the raw Excel dataset from disk.

    Args:
        path: Absolute path to the .xlsx dataset file.

    Returns:
        DataFrame containing the dataset records.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df: pd.DataFrame = pd.read_excel(path)
    logger.info("Loaded %d records from %s", len(df), path)

    required_columns = {"rep_note", "issue_category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}",
        )

    return df


def _validate_record(row: pd.Series, index: int) -> bool:
    """Validate that a record has the required fields.

    Args:
        row: A single dataset row.
        index: Row index for logging.

    Returns:
        True if the record is valid.
    """
    rep_note = row.get("rep_note", "")
    issue_category = row.get("issue_category", "")

    if not isinstance(rep_note, str) or not rep_note.strip():
        logger.warning("Row %d: missing or empty 'rep_note', skipping", index)
        return False

    if issue_category not in CATEGORY_MAP:
        logger.warning(
            "Row %d: unknown issue_category '%s', skipping",
            index,
            issue_category,
        )
        return False

    return True


def _map_category(uppercase_category: str) -> str:
    """Map an uppercase dataset category to its normalized name.

    Args:
        uppercase_category: Category value from the dataset.

    Returns:
        Normalized category name.
    """
    return CATEGORY_MAP.get(uppercase_category, uppercase_category.lower())


def prepare_dataset(
    dataset_path: str = config.dataset_path,
    test_size: float = config.test_size,
    random_seed: int = config.random_seed,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and test DataFrames from the Excel dataset.

    Reads the standardized sales dataset, extracts rep_note and
    issue_category, maps categories to clean names, and splits
    80/20 into train/test sets with stratification.

    Args:
        dataset_path: Path to the source Excel dataset.
        test_size: Fraction of data for testing (0.0 to 1.0).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df) with columns:
            'input' (str), 'output' (str).
    """
    raw_df: pd.DataFrame = _load_source_data(dataset_path)

    # Validate and filter records
    valid_mask = raw_df.apply(
        lambda row: _validate_record(row, row.name), axis=1,
    )
    valid_df = raw_df[valid_mask].copy()
    skipped = len(raw_df) - len(valid_df)
    logger.info(
        "Valid records: %d, skipped: %d", len(valid_df), skipped,
    )

    if len(valid_df) < 10:
        raise ValueError(
            f"Too few valid records ({len(valid_df)}) to split. "
            f"Check the dataset.",
        )

    # Map categories to normalized names
    valid_df["input"] = valid_df["rep_note"].str.strip()
    valid_df["output"] = valid_df["issue_category"].apply(_map_category)

    # Data leakage check: detect duplicate notes
    duplicates = valid_df["input"].duplicated().sum()
    if duplicates > 0:
        logger.warning(
            "⚠ Found %d duplicate notes in dataset. "
            "This can cause data leakage between train/test splits. "
            "Consider deduplication.", duplicates,
        )

    # Log category distribution
    category_counts = valid_df["output"].value_counts().to_dict()
    logger.info("Category distribution: %s", category_counts)

    # Stratified train/test split
    train_df, test_df = train_test_split(
        valid_df[["input", "output"]],
        test_size=test_size,
        random_state=random_seed,
        stratify=valid_df["output"],
    )

    logger.info("Train set: %d records", len(train_df))
    logger.info("Test set: %d records", len(test_df))

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_df, test_df = prepare_dataset()
    print(f"\nDataset preparation complete!")
    print(f"Train: {len(train_df)} records")
    print(f"Test: {len(test_df)} records")
    print(f"\nTrain distribution:\n{train_df['output'].value_counts()}")
