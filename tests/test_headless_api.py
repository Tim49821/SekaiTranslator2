import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from headless_server import TranslationOutcome, create_app


AUTH_HEADERS = {'Authorization': 'Bearer local-token'}


def png_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (1, 1), color).save(buffer, format='PNG')
    return buffer.getvalue()


class FakeBridge:
    busy = False

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def translate(self, project_dir: str) -> TranslationOutcome:
        self.calls.append(project_dir)
        if self.fail:
            return TranslationOutcome(False, status_code=503, error='translator unavailable')
        result_path = Path(project_dir) / 'result.png'
        result_path.write_bytes(png_bytes((0, 255, 0)))
        return TranslationOutcome(
            True,
            result_path=str(result_path),
            media_type='image/png',
        )


class HeadlessApiTest(unittest.TestCase):
    def make_app(self, bridge=None):
        bridge = bridge or FakeBridge()
        return create_app(
            bridge,
            threading.Lock(),
            api_token='local-token',
            result_ttl_seconds=60,
            max_upload_bytes=1024 * 1024,
        )

    def test_sync_translate_requires_auth_and_returns_image(self):
        bridge = FakeBridge()
        app = self.make_app(bridge)
        with TestClient(app) as client:
            unauthorized = client.post(
                '/translate',
                files={'file': ('input.png', png_bytes(), 'image/png')},
            )
            self.assertEqual(unauthorized.status_code, 401)

            translated = client.post(
                '/translate',
                headers=AUTH_HEADERS,
                files={'file': ('input.png', png_bytes(), 'image/png')},
            )

            self.assertEqual(translated.status_code, 200, translated.text)
            self.assertEqual(translated.headers['content-type'], 'image/png')
            self.assertEqual(translated.content, png_bytes((0, 255, 0)))
            self.assertEqual(len(bridge.calls), 1)

    def test_async_job_flow_reports_status_and_result(self):
        app = self.make_app()
        with TestClient(app) as client:
            created = client.post(
                '/jobs',
                headers=AUTH_HEADERS,
                files={'file': ('input.png', png_bytes(), 'image/png')},
            )
            self.assertEqual(created.status_code, 202, created.text)
            job_id = created.json()['job_id']

            status = None
            for _ in range(100):
                response = client.get(f'/jobs/{job_id}', headers=AUTH_HEADERS)
                status = response.json()['status']
                if status in {'done', 'failed'}:
                    break
                time.sleep(0.01)

            self.assertEqual(status, 'done')
            result = client.get(f'/jobs/{job_id}/result', headers=AUTH_HEADERS)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.content, png_bytes((0, 255, 0)))

    def test_async_failure_preserves_status_code_and_message(self):
        app = self.make_app(FakeBridge(fail=True))
        with TestClient(app) as client:
            created = client.post(
                '/jobs',
                headers=AUTH_HEADERS,
                files={'file': ('input.png', png_bytes(), 'image/png')},
            )
            job_id = created.json()['job_id']

            status = None
            for _ in range(100):
                status_response = client.get(f'/jobs/{job_id}', headers=AUTH_HEADERS)
                status = status_response.json()['status']
                if status == 'failed':
                    break
                time.sleep(0.01)

            self.assertEqual(status, 'failed')
            result = client.get(f'/jobs/{job_id}/result', headers=AUTH_HEADERS)
            self.assertEqual(result.status_code, 503)
            self.assertIn('translator unavailable', result.text)

    def test_invalid_image_is_rejected(self):
        app = self.make_app()
        with TestClient(app) as client:
            response = client.post(
                '/translate',
                headers=AUTH_HEADERS,
                files={'file': ('input.png', b'not-an-image', 'image/png')},
            )

            self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
