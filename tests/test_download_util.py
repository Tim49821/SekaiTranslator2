import os
import sys
import tempfile
import unittest
from unittest.mock import patch

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import download_util


class FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_content(self, chunk_size):
        return iter(self._chunks)


class DownloadUtilTest(unittest.TestCase):
    def test_download_ssl_context_is_verified_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(download_util._download_ssl_context())

    def test_download_ssl_context_allows_explicit_insecure_override(self):
        with patch.dict(os.environ, {download_util.INSECURE_DOWNLOAD_ENV: "1"}):
            self.assertIsNotNone(download_util._download_ssl_context())

    def test_save_response_content_moves_complete_partial_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = os.path.join(temp_dir, "model.bin")

            download_util.save_response_content(
                FakeResponse([b"abc", b"", b"def"]),
                target,
                file_size=None,
                chunk_size=3,
            )

            with open(target, "rb") as f:
                self.assertEqual(f.read(), b"abcdef")
            self.assertFalse([name for name in os.listdir(temp_dir) if name.endswith(".partial")])


if __name__ == "__main__":
    unittest.main()
