"""Enterprise-grade structured logging configuration for the Sales Intelligence System.

Provides:
- Structured JSON logging for production
- Correlation ID injection for request tracing
- Sensitive data redaction
- Consistent log format across all modules
- Console and file handlers with rotation
"""

import logging
import logging.handlers
import os
import re
import sys
from typing import Any, Dict, List, Optional


class SensitiveDataFilter(logging.Filter):
    """Redacts sensitive information from log records.

    Filters out API keys, tokens, and other secrets from log output
    to comply with security policies.
    """

    PATTERNS: List[re.Pattern] = [
        re.compile(r'(API_KEY|api-key|api_key|X-API-Key)\s*[:=]\s*\S+', re.IGNORECASE),
        re.compile(r'(password|passwd|pwd|secret|token)\s*[:=]\s*\S+', re.IGNORECASE),
        re.compile(r'(Authorization:\s*Bearer)\s+\S+', re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log message."""
        if isinstance(record.msg, str):
            for pattern in self.PATTERNS:
                record.msg = pattern.sub(r'\1=***REDACTED***', record.msg)
        if record.args:
            new_args: List[str] = []
            for arg in (record.args if isinstance(record.args, tuple) else [record.args]):
                if isinstance(arg, str):
                    for pattern in self.PATTERNS:
                        arg = pattern.sub(r'\1=***REDACTED***', arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        return True


class CorrelationIdFilter(logging.Filter):
    """Injects correlation ID into log records.

    Reads the correlation ID from thread-local storage or context,
    falling back to 'N/A' when no correlation ID is set.
    """

    _correlation_id: Optional[str] = None

    @classmethod
    def set_correlation_id(cls, correlation_id: Optional[str]) -> None:
        """Set the current thread's correlation ID."""
        cls._correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get the current thread's correlation ID."""
        return cls._correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self._correlation_id or "N/A"
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure application-wide logging.

    Must be called once at application startup.
    Subsequent calls will replace the existing configuration.

    Args:
        log_level: Python logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs to console only.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.
    """
    level: int = getattr(logging, (log_level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    fmt_string: str = (
        "%(asctime)s | %(levelname)-8s | %(correlation_id)s | "
        "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Build handlers
    handlers: List[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=fmt_string, datefmt=date_format))
    handlers.append(console_handler)

    # File handler (optional, with rotation)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt=fmt_string, datefmt=date_format))
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers with filters
    for handler in handlers:
        handler.addFilter(SensitiveDataFilter())
        handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
