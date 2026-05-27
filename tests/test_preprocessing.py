"""Comprehensive tests for the text preprocessing service.

Tests all preprocessing pipeline steps including edge cases,
abbreviation expansion, and error handling.
"""

import pytest

from app.core.exceptions import EmptyInputError
from app.services.preprocessing import TextPreprocessor


class TestToLower:
    """Tests for lowercase conversion."""

    def test_basic_lowercase(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("Hello World") == "hello world"

    def test_mixed_case(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("ReTaIlErS") == "retailers"

    def test_already_lower(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("already lower") == "already lower"

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("") == ""

    def test_numbers_and_letters(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("STORE 123 ABC") == "store 123 abc"


class TestCleanWhitespace:
    """Tests for whitespace cleanup."""

    def test_extra_spaces(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("  hello   world  ") == "hello world"

    def test_newlines(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello\nworld") == "hello world"

    def test_tabs(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello\tworld") == "hello world"

    def test_multiple_whitespace_types(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello \n\t world") == "hello world"

    def test_already_clean(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello world") == "hello world"

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("") == ""


class TestCleanPunctuation:
    """Tests for punctuation cleanup."""

    def test_exclamation_reduction(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_whitespace(preprocessor.clean_punctuation("hello!!!"))
        assert "!!!" not in result

    def test_question_reduction(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello???")
        assert "???" not in result

    def test_period_reduction(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello...")
        assert "..." not in result

    def test_comma_reduction(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello,,,")
        assert ",,," not in result

    def test_special_chars_removed(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello@world#test")
        assert "@" not in result
        assert "#" not in result

    def test_apostrophe_preserved(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("retailer's complaint")
        assert "'" in result

    def test_hyphen_removed(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("supply-chain")
        assert "-" not in result

    def test_dollar_sign_removed(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("$100 price")
        assert "$" not in result


class TestNormalizeTypos:
    """Tests for abbreviation normalization."""

    def test_stock_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "stock" in preprocessor.normalize_typos("stk low")

    def test_not_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "not" in preprocessor.normalize_typos("nt available")

    def test_coming_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "coming" in preprocessor.normalize_typos("cmng soon")

    def test_delivery_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "delivery" in preprocessor.normalize_typos("dlvry late")

    def test_retailer_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "retailer" in preprocessor.normalize_typos("ret unhappy")

    def test_price_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "price" in preprocessor.normalize_typos("pr too high")

    def test_quantity_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "quantity" in preprocessor.normalize_typos("qty low")

    def test_unknown_word_unchanged(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.normalize_typos("unknownword")
        assert "unknownword" in result

    def test_multiple_abbreviations(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.normalize_typos("stk nt cmng frm 3 dys")
        assert "stock" in result
        assert "not" in result
        assert "coming" in result
        assert "days" in result

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.normalize_typos("")
        assert result == ""


class TestFullPreprocess:
    """Tests for the full preprocessing pipeline."""

    def test_clean_note(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("Retailers Complaining About Delay")
        assert "retailers" in result
        assert "complaining" in result
        assert result == result.lower()

    def test_noisy_note_with_abbreviations(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("stk nt cmng frm 3 dys")
        assert "stock" in result
        assert "not" in result
        assert "coming" in result
        assert "days" in result

    def test_empty_raises_error(self, preprocessor: TextPreprocessor) -> None:
        with pytest.raises(EmptyInputError):
            preprocessor.preprocess("")

    def test_whitespace_only_raises_error(self, preprocessor: TextPreprocessor) -> None:
        with pytest.raises(EmptyInputError):
            preprocessor.preprocess("   ")

    def test_single_word(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("delay")
        assert result == "delay"

    def test_case_normalization(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("UPPERCASE TEXT")
        assert result == result.lower()

    def test_comprehensive_noise(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("  STK!!! nt cmng...  ")
        assert "stock" in result
        assert "not" in result
        assert "coming" in result

    def test_preserves_meaningful_content(
        self, preprocessor: TextPreprocessor,
    ) -> None:
        note: str = "Retailer reported stock shortage"
        result: str = preprocessor.preprocess(note)
        assert "retailer" in result
        assert "reported" in result
        assert "stock" in result
        assert "shortage" in result

    def test_output_is_stripped(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("  clean note  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_pipeline_idempotency(self, preprocessor: TextPreprocessor) -> None:
        note: str = "clean note"
        result1: str = preprocessor.preprocess(note)
        result2: str = preprocessor.preprocess(result1)
        assert result1 == result2
