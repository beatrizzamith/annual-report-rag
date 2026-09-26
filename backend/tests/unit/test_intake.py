"""Tests for validating and storing an uploaded PDF."""

import hashlib
import io
from pathlib import Path

import pytest

from app.core.errors import FileTooLargeError, InvalidFileError
from app.ingestion.intake import store_pdf, stream_to_temp_file

PDF_BYTES = b"%PDF-1.7\nsome report content"


def test_a_valid_pdf_is_saved_with_its_sha256(tmp_path: Path):
    temp_path, sha256 = stream_to_temp_file(io.BytesIO(PDF_BYTES), tmp_path, max_bytes=1000)

    assert temp_path.read_bytes() == PDF_BYTES
    assert sha256 == hashlib.sha256(PDF_BYTES).hexdigest()


def test_a_file_without_the_pdf_signature_is_rejected_and_leaves_nothing_behind(tmp_path: Path):
    with pytest.raises(InvalidFileError):
        stream_to_temp_file(io.BytesIO(b"<html>not a pdf</html>"), tmp_path, max_bytes=1000)

    assert list(tmp_path.iterdir()) == []


def test_an_empty_upload_is_rejected(tmp_path: Path):
    with pytest.raises(InvalidFileError):
        stream_to_temp_file(io.BytesIO(b""), tmp_path, max_bytes=1000)

    assert list(tmp_path.iterdir()) == []


def test_an_upload_over_the_size_limit_is_rejected_and_leaves_nothing_behind(tmp_path: Path):
    with pytest.raises(FileTooLargeError):
        stream_to_temp_file(io.BytesIO(PDF_BYTES), tmp_path, max_bytes=10)

    assert list(tmp_path.iterdir()) == []


def test_a_pdf_is_stored_under_its_hash_and_a_repeat_upload_keeps_the_first_copy(tmp_path: Path):
    pdfs_dir = tmp_path / "pdfs"
    first_temp, sha256 = stream_to_temp_file(io.BytesIO(PDF_BYTES), tmp_path, max_bytes=1000)
    stored_path = store_pdf(first_temp, sha256, pdfs_dir)

    second_temp, _ = stream_to_temp_file(io.BytesIO(PDF_BYTES), tmp_path, max_bytes=1000)
    repeat_path = store_pdf(second_temp, sha256, pdfs_dir)

    assert stored_path == pdfs_dir / f"{sha256}.pdf"
    assert repeat_path == stored_path
    assert not second_temp.exists()
