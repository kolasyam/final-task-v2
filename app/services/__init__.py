"""Services package.

Exports key service classes and constants for convenient importing.
"""

# CATEGORY_MAP and SUPPORTED_CATEGORIES are now in app.core.constants
# Re-export here for backward compatibility with existing imports
from app.core.constants import CATEGORY_MAP, SUPPORTED_CATEGORIES

from app.services.predictor import SalesNotePredictor
from app.services.preprocessing import TextPreprocessor
from app.services.storage import PredictionStorage

__all__ = [
    "CATEGORY_MAP",
    "SUPPORTED_CATEGORIES",
    "SalesNotePredictor",
    "TextPreprocessor",
    "PredictionStorage",
]
