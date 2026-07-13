import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from relay_server import (
    InvalidJobTransition,
    RelayCapacityExceeded,
    RelayJobStore,
)


def png_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (1, 1), color).save(buffer, format='PNG')
    return buffer.getvalue()


class RelayJobStoreTest(unittest.TestCase):
    def test_jobs_survive_store_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(temp_dir, 60, 1024 * 1024) as store:
                submitted = store.submit_bytes(png_bytes(), 'input.png', '.png')
                job_id = submitted.job_id

            with RelayJobStore(temp_dir, 60, 1024 * 1024) as reopened:
                restored = reopened.get(job_id)

                self.assertIsNotNone(restored)
                self.assertEqual(restored.status, 'queued')
                self.assertTrue(Path(restored.input_path).is_file())

    def test_expired_worker_lease_is_requeued_after_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = time.time()
            with RelayJobStore(temp_dir, 60, 1024 * 1024, claim_lease_seconds=1) as store:
                job = store.submit_bytes(png_bytes(), 'input.png', '.png')
                with patch('relay_server.time.time', return_value=now):
                    store.claim_next('worker-a')

            with patch('relay_server.time.time', return_value=now + 2):
                with RelayJobStore(temp_dir, 60, 1024 * 1024, claim_lease_seconds=1) as reopened:
                    restored = reopened.get(job.job_id)

                    self.assertEqual(restored.status, 'queued')
                    self.assertIsNone(restored.worker_id)
                    self.assertIsNone(restored.lease_expires_at)

    def test_only_claim_owner_can_heartbeat_or_finish_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(temp_dir, 60, 1024 * 1024) as store:
                job = store.submit_bytes(png_bytes(), 'input.png', '.png')
                claimed = store.claim_next('worker-a')

                self.assertEqual(claimed.job_id, job.job_id)
                with self.assertRaises(InvalidJobTransition):
                    store.heartbeat(job.job_id, 'worker-b')
                with self.assertRaises(InvalidJobTransition):
                    store.fail(job.job_id, 'wrong worker', 'worker-b')

                failed = store.fail(job.job_id, 'translation failed', 'worker-a')
                self.assertEqual(failed.status, 'failed')
                with self.assertRaises(InvalidJobTransition):
                    store.fail(job.job_id, 'duplicate', 'worker-a')

    def test_expired_lease_cannot_be_completed_or_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = time.time()
            with RelayJobStore(temp_dir, 60, 1024 * 1024, claim_lease_seconds=1) as store:
                completed_job = store.submit_bytes(png_bytes(), 'complete.png', '.png')
                failed_job = store.submit_bytes(png_bytes(), 'fail.png', '.png')
                with patch('relay_server.time.time', return_value=now):
                    store.claim_next('worker-a')
                    store.claim_next('worker-a')

                with patch('relay_server.time.time', return_value=now + 2):
                    with self.assertRaises(InvalidJobTransition):
                        store.complete(
                            completed_job.job_id,
                            str(Path(completed_job.job_dir) / 'result.png'),
                            'image/png',
                            'worker-a',
                        )

                    with self.assertRaises(InvalidJobTransition):
                        store.fail(failed_job.job_id, 'too late', 'worker-a')

                    reclaimed = store.claim_next('worker-b')
                    self.assertEqual(reclaimed.job_id, completed_job.job_id)

                self.assertEqual(store.get(completed_job.job_id).status, 'running')
                self.assertEqual(store.get(completed_job.job_id).worker_id, 'worker-b')
                self.assertEqual(store.get(failed_job.job_id).status, 'queued')

    def test_job_count_limit_rejects_new_work_and_cleans_partial_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(temp_dir, 60, 1024 * 1024, max_jobs=1) as store:
                store.submit_bytes(png_bytes(), 'first.png', '.png')

                with self.assertRaises(RelayCapacityExceeded):
                    store.submit_bytes(png_bytes((0, 255, 0)), 'second.png', '.png')

                job_dirs = [path for path in Path(temp_dir).iterdir() if path.is_dir()]
                self.assertEqual(len(job_dirs), 1)

    def test_storage_limit_rejects_job_that_would_exceed_quota(self):
        image = png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            with RelayJobStore(
                temp_dir,
                60,
                max_upload_bytes=len(image) + 1,
                max_jobs=10,
                max_storage_bytes=len(image) + 1,
            ) as store:
                store.submit_bytes(image, 'first.png', '.png')

                with self.assertRaises(RelayCapacityExceeded):
                    store.submit_bytes(image, 'second.png', '.png')

                self.assertEqual(store.stats()['total'], 1)


if __name__ == '__main__':
    unittest.main()
