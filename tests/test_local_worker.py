import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_worker import RelayWorker


class LocalWorkerProtocolTest(unittest.TestCase):
    def make_worker(self, heartbeat_interval=0):
        return RelayWorker(
            relay_url='http://relay.test',
            local_url='http://local.test',
            worker_token='worker-token',
            local_token='local-token',
            worker_id='worker-a',
            poll_interval=0.1,
            request_timeout=10,
            heartbeat_interval=heartbeat_interval,
        )

    def test_claim_includes_worker_identity(self):
        worker = self.make_worker()
        response = MagicMock(status_code=204)
        worker.session.get = MagicMock(return_value=response)

        self.assertIsNone(worker.claim_job())
        self.assertEqual(worker.session.get.call_args.kwargs['params'], {'worker_id': 'worker-a'})

    def test_worker_identity_is_sent_for_input_result_and_failure(self):
        worker = self.make_worker()
        response = MagicMock(status_code=200)
        response.iter_content.return_value = [b'image-bytes']
        worker.session.get = MagicMock(return_value=response)
        worker.session.post = MagicMock(return_value=response)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = worker.download_input(
                {'input_url': '/worker/jobs/job-1/input', 'input_filename': 'input.png'},
                temp_dir,
            )
            result_path = Path(temp_dir) / 'result.png'
            result_path.write_bytes(b'result-bytes')
            worker.upload_result('job-1', result_path, 'image/png')
            worker.report_failure('job-1', 'failed')

        self.assertEqual(input_path.name, 'input.png')
        self.assertEqual(worker.session.get.call_args.kwargs['params'], {'worker_id': 'worker-a'})
        for call in worker.session.post.call_args_list:
            self.assertEqual(call.kwargs['params'], {'worker_id': 'worker-a'})

    def test_heartbeat_renews_lease_while_work_is_running(self):
        worker = self.make_worker(heartbeat_interval=0.01)
        heartbeat_session = MagicMock()
        heartbeat_session.post.return_value = MagicMock(status_code=200)
        session_context = MagicMock()
        session_context.__enter__.return_value = heartbeat_session

        with patch('local_worker.requests.Session', return_value=session_context):
            with worker.maintain_lease('job-1'):
                time.sleep(0.04)

        self.assertGreaterEqual(heartbeat_session.post.call_count, 1)
        self.assertEqual(
            heartbeat_session.post.call_args.kwargs['params'],
            {'worker_id': 'worker-a'},
        )


if __name__ == '__main__':
    unittest.main()
