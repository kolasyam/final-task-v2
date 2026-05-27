"""Text preprocessing service for sales representative notes.

Provides a pipeline for normalizing and cleaning raw sales notes
before they are sent to inference backends.
"""

import logging
import re
from typing import Dict, List, Pattern

from app.core.constants import ABBREVIATION_MAP
from app.core.exceptions import EmptyInputError

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocesses sales notes for model input.

    Pipeline steps:
        1. Lowercase conversion
        2. Punctuation normalization
        3. Abbreviation expansion
        4. Whitespace cleanup

    All patterns and mappings are sourced from app.core.constants
    to maintain a single source of truth.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with compiled regex patterns."""
        self._abbreviation_map: Dict[str, str] = ABBREVIATION_MAP
        self._patterns: Dict[str, Pattern] = self._compile_patterns()
        logger.info("TextPreprocessor initialized")

    @staticmethod
    def _compile_patterns() -> Dict[str, Pattern]:
        """Compile regex patterns for reuse across method calls.

        Returns:
            Dictionary of compiled regex patterns.
        """
        return {
            "multi_exclamation": re.compile(r"[!]{2,}"),
            "multi_question": re.compile(r"[?]{2,}"),
            "multi_period": re.compile(r"[.]{2,}"),
            "multi_comma": re.compile(r"[,]{2,}"),
            "non_alphanumeric": re.compile(r"[^a-zA-Z0-9\s.,!?']"),
            "whitespace": re.compile(r"\s+"),
            "non_alpha": re.compile(r"[^a-zA-Z]"),
        }

    def to_lower(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
            text: Input text.

        Returns:
            Lowercased text.
        """
        return text.lower()

    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and strip.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        return self._patterns["whitespace"].sub(" ", text).strip()

    def clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation while preserving meaningful characters.

        Collapses repeated punctuation marks and removes special characters
        that are not part of standard text.

        Args:
            text: Input text.

        Returns:
            Text with cleaned punctuation.
        """
        text = self._patterns["multi_exclamation"].sub("!", text)
        text = self._patterns["multi_question"].sub("?", text)
        text = self._patterns["multi_period"].sub(".", text)
        text = self._patterns["multi_comma"].sub(",", text)
        text = self._patterns["non_alphanumeric"].sub("", text)
        return text

    def normalize_typos(self, text: str) -> str:
        """Normalize common typos and abbreviations.

        Uses the ABBREVIATION_MAP from app.core.constants to expand
        sales field abbreviations to their full forms.

        Args:
            text: Input text.

        Returns:
            Text with expanded abbreviations.
        """
        words: List[str] = text.split()
        normalized: List[str] = []
        non_alpha: Pattern = self._patterns["non_alpha"]

        for word in words:
            cleaned_word: str = non_alpha.sub("", word.lower())
            normalized.append(self._abbreviation_map.get(cleaned_word, word))

        return " ".join(normalized)

    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline to text.

        Args:
            text: Raw input text from sales representative note.

        Returns:
            Cleaned and normalized text ready for model input.

        Raises:
            EmptyInputError: If input text is empty or whitespace.
        """
        if not text or not text.strip():
            raise EmptyInputError("Input text")

        text = self.to_lower(text)
        text = self.clean_punctuation(text)
        text = self.normalize_typos(text)
        text = self.clean_whitespace(text)

        logger.debug("Preprocessed text: %s", text)
        return text
