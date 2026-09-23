"""Configures JSON logging for local app diagnostics."""

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Renders one log record as a single-line JSON string.

        Args:
            record: The log record to render.

        Returns:
            A single-line JSON string with timestamp, level, logger name,
            message, any `extra_fields` passed via `log_event`, and a
            formatted traceback when the record carries exception info.
        """
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Replaces the root logger's handlers with one JSON stdout handler.

    Args:
        level: The minimum level to log, for example "INFO" or "DEBUG".
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Logs an info-level message with structured fields attached.

    Args:
        logger: The logger to log through.
        message: The human-readable log message.
        **fields: Structured fields rendered alongside `message` by
            `JsonFormatter`, for example `report_id=1` or `stage="parse"`.
    """
    logger.info(message, extra={"extra_fields": fields})
