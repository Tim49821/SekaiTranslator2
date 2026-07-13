import io
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from relay_server import create_app


CLIENT_HEADERS = {'Authorization': 'Bearer client-token'}
WORKER_HEADERS = {'Authorization': 'Bearer worker-token'}


def png_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (1, 1), color).save(buffer, format='PNG')
    return buffer.getvalue()


class RelayApiTest(unittest.TestCase):
    def make_app(self, storage_dir: str, **kwargs):
        return create_app(
            storage_dir,
            api_token='client-token',
            worker_token='worker-token',
            result_ttl_seconds=60,
            max_upload_bytes=1024 * 1024,
            claim_lease_seconds=60,
            **kwargs,
        )

    def test_authenticated_job_worker_and_result_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir)
            with TestClient(app) as client:
                unauthorized = client.post(
                    '/jobs/raw?filename=input.png',
                    content=png_bytes(),
                    headers={'Content-Type': 'image/png'},
                )
                self.assertEqual(unauthorized.status_code, 401)

                created = client.post(
                    '/jobs/raw?filename=input.png',
                    content=png_bytes(),
                    headers={**CLIENT_HEADERS, 'Content-Type': 'image/png'},
                )
                self.assertEqual(created.status_code, 202, created.text)
                job_id = created.json()['job_id']

                claimed = client.get(
                    '/worker/jobs/next?worker_id=worker-a',
                    headers=WORKER_HEADERS,
                )
                self.assertEqual(claimed.status_code, 200, claimed.text)
                self.assertEqual(claimed.json()['job_id'], job_id)
                self.assertIsNotNone(claimed.json()['lease_expires_at'])

                wrong_owner = client.get(
                    f'/worker/jobs/{job_id}/input?worker_id=worker-b',
                    headers=WORKER_HEADERS,
                )
                self.assertEqual(wrong_owner.status_code, 409)

                heartbeat = client.post(
                    f'/worker/jobs/{job_id}/heartbeat?worker_id=worker-a',
                    headers=WORKER_HEADERS,
                )
                self.assertEqual(heartbeat.status_code, 200, heartbeat.text)

                worker_input = client.get(
                    f'/worker/jobs/{job_id}/input?worker_id=worker-a',
                    headers=WORKER_HEADERS,
                )
                self.assertEqual(worker_input.status_code, 200)
                self.assertEqual(worker_input.content, png_bytes())

                result_bytes = png_bytes((0, 255, 0))
                completed = client.post(
                    f'/worker/jobs/{job_id}/result?worker_id=worker-a',
                    headers=WORKER_HEADERS,
                    files={'file': ('result.png', result_bytes, 'image/png')},
                )
                self.assertEqual(completed.status_code, 200, completed.text)
                self.assertEqual(completed.json()['status'], 'done')

                duplicate = client.post(
                    f'/worker/jobs/{job_id}/result?worker_id=worker-a',
                    headers=WORKER_HEADERS,
                    files={'file': ('result.png', result_bytes, 'image/png')},
                )
                self.assertEqual(duplicate.status_code, 409)

                result = client.get(f'/jobs/{job_id}/result', headers=CLIENT_HEADERS)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.content, result_bytes)

    def test_job_metadata_survives_app_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_app = self.make_app(temp_dir)
            with TestClient(first_app) as client:
                created = client.post(
                    '/jobs/raw?filename=input.png',
                    content=png_bytes(),
                    headers={**CLIENT_HEADERS, 'Content-Type': 'image/png'},
                )
                job_id = created.json()['job_id']

            second_app = self.make_app(temp_dir)
            with TestClient(second_app) as client:
                restored = client.get(f'/jobs/{job_id}', headers=CLIENT_HEADERS)

                self.assertEqual(restored.status_code, 200, restored.text)
                self.assertEqual(restored.json()['status'], 'queued')

    def test_capacity_limit_returns_service_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir, max_jobs=1)
            with TestClient(app) as client:
                first = client.post(
                    '/jobs/raw?filename=first.png',
                    content=png_bytes(),
                    headers={**CLIENT_HEADERS, 'Content-Type': 'image/png'},
                )
                second = client.post(
                    '/jobs/raw?filename=second.png',
                    content=png_bytes(),
                    headers={**CLIENT_HEADERS, 'Content-Type': 'image/png'},
                )

                self.assertEqual(first.status_code, 202, first.text)
                self.assertEqual(second.status_code, 503, second.text)


if __name__ == '__main__':
    unittest.main()
