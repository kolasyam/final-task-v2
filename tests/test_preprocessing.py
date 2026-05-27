"""Tests for the text preprocessing service."""

import pytest

from app.services.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Create a TextPreprocessor instance."""
    return TextPreprocessor()


class TestToLower:
    """Tests for to_lower conversion."""

    def test_basic_lowercase(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("Hello World") == "hello world"

    def test_mixed_case(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("ReTaIlErS") == "retailers"

    def test_already_lower(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.to_lower("already lower") == "already lower"


class TestCleanWhitespace:
    """Tests for whitespace cleanup."""

    def test_extra_spaces(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("  hello   world  ") == "hello world"

    def test_newlines(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello\nworld") == "hello world"

    def test_tabs(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_whitespace("hello\tworld") == "hello world"


class TestCleanPunctuation:
    """Tests for punctuation cleanup."""

    def test_exclamation(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello!!!")
        assert "!!!" not in result

    def test_special_chars(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.clean_punctuation("hello@world#test")
        assert "@" not in result
        assert "#" not in result


class TestNormalizeTypos:
    """Tests for abbreviation normalization."""

    def test_stock_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "stock" in preprocessor.normalize_typos("stk low")

    def test_not_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "not" in preprocessor.normalize_typos("nt available")

    def test_coming_abbreviation(self, preprocessor: TextPreprocessor) -> None:
        assert "coming" in preprocessor.normalize_typos("cmng soon")


class TestFullPreprocess:
    """Tests for the full preprocessing pipeline."""

    def test_clean_note(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("Retailers Complaining About Delay")
        assert "retailers" in result
        assert "complaining" in result

    def test_noisy_note(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("stk nt cmng frm 3 dys")
        assert "stock" in result
        assert "not" in result
        assert "coming" in result
        assert "days" in result

    def test_empty_raises_error(self, preprocessor: TextPreprocessor) -> None:
        with pytest.raises(ValueError, match="empty"):
            preprocessor.preprocess("")

    def test_whitespace_only_raises_error(self, preprocessor: TextPreprocessor) -> None:
        with pytest.raises(ValueError, match="empty"):
            preprocessor.preprocess("   ")

    def test_single_word(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("delay")
        assert result == "delay"

    def test_case_normalization(self, preprocessor: TextPreprocessor) -> None:
        result: str = preprocessor.preprocess("UPPERCASE TEXT")
        assert result == result.lower()
