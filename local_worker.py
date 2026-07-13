import argparse
import mimetypes
import os
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin

import requests


def bearer_headers(token: str) -> dict:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def normalize_base_url(url: str) -> str:
    return url.rstrip("/") + "/"


def response_error_text(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return str(body)[:2000]


class RelayWorker:
    def __init__(
        self,
        relay_url: str,
        local_url: str,
        worker_token: str,
        local_token: str,
        worker_id: str,
        poll_interval: float,
        request_timeout: float,
        heartbeat_interval: float = 60.0,
    ):
        self.relay_url = normalize_base_url(relay_url)
        self.local_url = normalize_base_url(local_url)
        self.worker_headers = bearer_headers(worker_token)
        self.local_headers = bearer_headers(local_token)
        self.worker_id = worker_id
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self.heartbeat_interval = max(heartbeat_interval, 0.0)
        self.session = requests.Session()

    def run_forever(self) -> None:
        while True:
            try:
                processed = self.process_once()
                if not processed:
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"worker error: {exc}")
                time.sleep(self.poll_interval)

    def process_once(self) -> bool:
        job = self.claim_job()
        if job is None:
            return False

        job_id = job["job_id"]
        try:
            with self.maintain_lease(job_id):
                with tempfile.TemporaryDirectory(prefix="balloontrans-worker-") as tmp_dir:
                    input_path = self.download_input(job, tmp_dir)
                    result_path, media_type = self.translate_local(input_path, job.get("input_filename") or input_path.name, tmp_dir)
                    self.upload_result(job_id, result_path, media_type)
                    print(f"completed job {job_id}")
        except Exception as exc:
            self.report_failure(job_id, str(exc))
            print(f"failed job {job_id}: {exc}")
        return True

    def claim_job(self):
        response = self.session.get(
            urljoin(self.relay_url, "worker/jobs/next"),
            params={"worker_id": self.worker_id},
            headers=self.worker_headers,
            timeout=self.request_timeout,
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()

    @contextmanager
    def maintain_lease(self, job_id: str):
        if self.heartbeat_interval <= 0:
            yield
            return

        stop_event = threading.Event()

        def heartbeat_loop():
            with requests.Session() as heartbeat_session:
                while not stop_event.wait(self.heartbeat_interval):
                    try:
                        response = heartbeat_session.post(
                            urljoin(self.relay_url, f"worker/jobs/{job_id}/heartbeat"),
                            params={"worker_id": self.worker_id},
                            headers=self.worker_headers,
                            timeout=min(self.request_timeout, 10.0),
                        )
                        response.raise_for_status()
                    except Exception as exc:
                        print(f"heartbeat failed for job {job_id}: {exc}")

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"relay-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=min(self.heartbeat_interval + 1.0, 5.0))

    def download_input(self, job: dict, tmp_dir: str) -> Path:
        input_url = urljoin(self.relay_url, job["input_url"].lstrip("/"))
        response = self.session.get(
            input_url,
            params={"worker_id": self.worker_id},
            headers=self.worker_headers,
            timeout=self.request_timeout,
            stream=True,
        )
        response.raise_for_status()

        filename = job.get("input_filename") or "input.png"
        suffix = Path(filename).suffix or ".png"
        input_path = Path(tmp_dir) / f"input{suffix}"
        with input_path.open("wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)
        if input_path.stat().st_size == 0:
            raise RuntimeError("downloaded input is empty")
        return input_path

    def translate_local(self, input_path: Path, filename: str, tmp_dir: str):
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with input_path.open("rb") as input_file:
            response = self.session.post(
                urljoin(self.local_url, "translate"),
                headers=self.local_headers,
                files={"file": (filename, input_file, content_type)},
                timeout=self.request_timeout,
            )
        if response.status_code != 200:
            raise RuntimeError(f"local translator returned {response.status_code}: {response_error_text(response)}")

        result_type = response.headers.get("content-type", "image/png").split(";")[0]
        suffix = mimetypes.guess_extension(result_type) or ".png"
        result_path = Path(tmp_dir) / f"result{suffix}"
        result_path.write_bytes(response.content)
        if result_path.stat().st_size == 0:
            raise RuntimeError("local translator returned an empty result")
        return result_path, result_type

    def upload_result(self, job_id: str, result_path: Path, media_type: str) -> None:
        with result_path.open("rb") as result_file:
            response = self.session.post(
                urljoin(self.relay_url, f"worker/jobs/{job_id}/result"),
                params={"worker_id": self.worker_id},
                headers=self.worker_headers,
                files={"file": (result_path.name, result_file, media_type)},
                timeout=self.request_timeout,
            )
        response.raise_for_status()

    def report_failure(self, job_id: str, error: str) -> None:
        try:
            response = self.session.post(
                urljoin(self.relay_url, f"worker/jobs/{job_id}/failed"),
                params={"worker_id": self.worker_id},
                headers=self.worker_headers,
                json={"status_code": 500, "error": error},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"failed to report job failure for {job_id}: {exc}")


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def parse_args():
    parser = argparse.ArgumentParser(description="BalloonsTranslator relay worker")
    parser.add_argument("--relay-url", default=os.environ.get("BALLOONTRANS_RELAY_URL", "http://127.0.0.1:9000"))
    parser.add_argument("--local-url", default=os.environ.get("BALLOONTRANS_LOCAL_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--worker-token", default=os.environ.get("BALLOONTRANS_RELAY_WORKER_TOKEN", os.environ.get("BALLOONTRANS_API_TOKEN", "")))
    parser.add_argument("--local-token", default=os.environ.get("BALLOONTRANS_LOCAL_API_TOKEN", os.environ.get("BALLOONTRANS_API_TOKEN", "")))
    parser.add_argument("--worker-id", default=os.environ.get("BALLOONTRANS_WORKER_ID", default_worker_id()))
    parser.add_argument("--poll-interval", default=2.0, type=float)
    parser.add_argument("--request-timeout", default=900.0, type=float)
    parser.add_argument("--heartbeat-interval", default=60.0, type=float)
    parser.add_argument("--once", action="store_true", help="process at most one job and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    worker = RelayWorker(
        relay_url=args.relay_url,
        local_url=args.local_url,
        worker_token=args.worker_token,
        local_token=args.local_token,
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        request_timeout=args.request_timeout,
        heartbeat_interval=args.heartbeat_interval,
    )
    if args.once:
        worker.process_once()
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
