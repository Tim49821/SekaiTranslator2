import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from headless_server import save_upload_to_project
from relay_server import RelayJobStore
from utils.api_uploads import InvalidImageUpload, UploadTooLarge, copy_upload_file, validate_image_file


class FakeUpload:
    def __init__(self, content: bytes):
        self.file = io.BytesIO(content)
        self.filename = "input.png"
        self.content_type = "image/png"


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class ApiUploadTest(unittest.TestCase):
    def test_copy_upload_file_rejects_oversized_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "input.png"

            with self.assertRaises(UploadTooLarge):
                copy_upload_file(io.BytesIO(b"abcdef"), target, max_bytes=3)

    def test_validate_image_file_rejects_non_image_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "input.png"
            with open(target, "wb") as f:
                f.write(b"not an image")

            with self.assertRaises(InvalidImageUpload):
                validate_image_file(target)

    def test_relay_store_validates_raw_job_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(temp_dir, result_ttl_seconds=60, max_upload_bytes=1024 * 1024) as store:

                job = store.submit_bytes(png_bytes(), "input.png", ".png")

                self.assertEqual(job.status, "queued")
                self.assertTrue(os.path.exists(job.input_path))

    def test_relay_store_rejects_invalid_raw_image_and_cleans_job_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(temp_dir, result_ttl_seconds=60, max_upload_bytes=1024 * 1024) as store:

                with self.assertRaises(InvalidImageUpload):
                    store.submit_bytes(b"not an image", "input.png", ".png")

                self.assertEqual(os.listdir(temp_dir), [RelayJobStore.DB_FILENAME])

    def test_headless_save_upload_to_project_validates_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_upload_to_project(FakeUpload(png_bytes()), ".png", temp_dir, max_upload_bytes=1024 * 1024)

            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
