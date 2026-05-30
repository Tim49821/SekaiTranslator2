import os
from pathlib import Path


DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES_ENV = "BALLOONTRANS_MAX_UPLOAD_BYTES"
READ_CHUNK_SIZE = 1024 * 1024


class UploadTooLarge(ValueError):
    pass


class InvalidImageUpload(ValueError):
    pass


def max_upload_bytes_from_env(default: int = DEFAULT_MAX_UPLOAD_BYTES) -> int:
    raw_value = os.environ.get(MAX_UPLOAD_BYTES_ENV, "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return default


def upload_too_large_message(max_bytes: int) -> str:
    size_mb = max_bytes / (1024 * 1024)
    return f"Uploaded file exceeds the {size_mb:.1f} MB limit."


def copy_upload_file(file_obj, destination: Path, max_bytes: int) -> int:
    total = 0
    with destination.open("wb") as out_file:
        while True:
            chunk = file_obj.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge(upload_too_large_message(max_bytes))
            out_file.write(chunk)
    if total == 0:
        raise ValueError("Uploaded file is empty.")
    return total


def write_upload_bytes(content: bytes, destination: Path, max_bytes: int) -> int:
    total = len(content)
    if total > max_bytes:
        raise UploadTooLarge(upload_too_large_message(max_bytes))
    destination.write_bytes(content)
    if total == 0:
        raise ValueError("Uploaded file is empty.")
    return total


def validate_image_file(path: Path) -> None:
    try:
        try:
            import pillow_jxl  # noqa: F401
        except Exception:
            pass
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise InvalidImageUpload("Uploaded file is not a valid image.") from exc
