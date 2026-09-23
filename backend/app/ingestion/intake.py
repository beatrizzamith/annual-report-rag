"""Validates uploaded PDFs, hashes them, and stores them once per unique file.

This module is deliberately decoupled from FastAPI's `UploadFile`: any object
with a synchronous `.read(size)` method works, which keeps it easy to test.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.config import Settings
from app.core.errors import FileTooLargeError, InvalidFileError
from app.db.models import Report
from app.db.repositories import ReportsRepo

_CHUNK_SIZE = 1024 * 1024
_PDF_MAGIC = b"%PDF-"


class ReadableStream(Protocol):
    """Anything that can be read in chunks, e.g. an `UploadFile.file`."""

    def read(self, size: int = -1) -> bytes:
        """Reads up to `size` bytes from the stream.

        Args:
            size: The maximum number of bytes to read. Reads to end of
                stream if negative.

        Returns:
            Up to `size` bytes, or an empty `bytes` at end of stream.
        """
        ...


def stream_to_temp_file(stream: ReadableStream, tmp_dir: Path, max_bytes: int) -> tuple[Path, str]:
    """Streams `stream` to a temp file while computing its SHA-256.

    Rejects non-PDF content and oversized uploads without buffering the
    whole file in memory.

    Args:
        stream: The upload's readable stream.
        tmp_dir: Directory to write the temp file into (created if missing).
        max_bytes: The maximum allowed upload size, in bytes.

    Returns:
        A tuple of the temp file's path and its SHA-256 hex digest.

    Raises:
        InvalidFileError: The stream is empty or does not start with the
            PDF magic bytes.
        FileTooLargeError: The stream exceeds `max_bytes`.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload-{uuid4().hex}.tmp"
    hasher = hashlib.sha256()
    total = 0
    seen_any = False

    with open(tmp_path, "wb") as handle:
        while True:
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            if not seen_any:
                if not chunk.startswith(_PDF_MAGIC):
                    tmp_path.unlink(missing_ok=True)
                    raise InvalidFileError("File does not look like a PDF (missing %PDF- header)")
                seen_any = True
            total += len(chunk)
            if total > max_bytes:
                tmp_path.unlink(missing_ok=True)
                raise FileTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
            hasher.update(chunk)
            handle.write(chunk)

    if not seen_any:
        tmp_path.unlink(missing_ok=True)
        raise InvalidFileError("Uploaded file is empty")

    return tmp_path, hasher.hexdigest()


def store_pdf(tmp_path: Path, sha256: str, pdfs_dir: Path) -> Path:
    """Moves a validated upload into permanent storage, named by hash.

    Args:
        tmp_path: The temp file produced by `stream_to_temp_file`.
        sha256: The file's SHA-256 hex digest.
        pdfs_dir: Directory PDFs are stored in (created if missing).

    Returns:
        The file's final path, `pdfs_dir/{sha256}.pdf`. If that path
        already exists (the same file was uploaded before), `tmp_path` is
        discarded instead of overwriting it.
    """
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    final_path = pdfs_dir / f"{sha256}.pdf"
    if final_path.exists():
        tmp_path.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_path), str(final_path))
    return final_path


@dataclass
class IntakeResult:
    """The outcome of `intake_upload`."""

    report: Report
    needs_ingestion: bool  # False when an existing `ready` report was returned as-is


def intake_upload(
    stream: ReadableStream,
    filename: str,
    company: str,
    fiscal_year: int,
    settings: Settings,
    reports_repo: ReportsRepo,
) -> IntakeResult:
    """Validates, stores and deduplicates one uploaded report.

    Args:
        stream: The upload's readable stream.
        filename: The original filename, for display.
        company: The company name typed at upload.
        fiscal_year: The fiscal year typed at upload.
        settings: Application settings (upload size limit, storage paths).
        reports_repo: Repository for reading and writing the `reports` table.

    Returns:
        The report (existing or newly created) and whether the caller still
        needs to run ingestion for it: False when an already-`ready` or
        already-`processing` report with the same content was found.

    Raises:
        InvalidFileError: The upload is empty or not a PDF.
        FileTooLargeError: The upload exceeds `settings.max_upload_mb`.
    """
    tmp_dir = settings.data_dir / "tmp"
    tmp_path, sha256 = stream_to_temp_file(stream, tmp_dir, settings.max_upload_mb * 1024 * 1024)

    existing = reports_repo.get_by_sha256(sha256)

    if existing and existing.status == "ready":
        tmp_path.unlink(missing_ok=True)
        return IntakeResult(report=existing, needs_ingestion=False)

    if existing and existing.status == "processing":
        tmp_path.unlink(missing_ok=True)
        return IntakeResult(report=existing, needs_ingestion=False)

    store_pdf(tmp_path, sha256, settings.pdfs_dir)

    if existing and existing.status == "failed":
        reports_repo.reset_for_retry(existing.id)
        refreshed = reports_repo.get(existing.id)
        assert refreshed is not None
        return IntakeResult(report=refreshed, needs_ingestion=True)

    report = reports_repo.create(sha256, filename, company, fiscal_year)
    return IntakeResult(report=report, needs_ingestion=True)
