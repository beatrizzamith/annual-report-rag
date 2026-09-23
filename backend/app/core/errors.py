"""Typed application errors mapped to client-safe JSON responses."""


class AppError(Exception):
    """Base class for errors an exception handler maps to a JSON error body."""

    error_code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str):
        """Stores the human-readable message returned to the client.

        Args:
            message: A human-readable description, returned to the client.
        """
        super().__init__(message)
        self.message = message


class InvalidFileError(AppError):
    error_code = "invalid_file"
    status_code = 400


class FileTooLargeError(AppError):
    error_code = "file_too_large"
    status_code = 413


class LLMUnavailableError(AppError):
    error_code = "llm_unavailable"
    status_code = 503


class LLMRateLimitedError(AppError):
    error_code = "llm_rate_limited"
    status_code = 429


class LLMAuthFailedError(AppError):
    error_code = "llm_auth_failed"
    status_code = 401


class NoReportsLoadedError(AppError):
    error_code = "no_reports_loaded"
    status_code = 200  # not a failure; the chat endpoint returns this as a status


class ReportNotFoundError(AppError):
    error_code = "report_not_found"
    status_code = 404
