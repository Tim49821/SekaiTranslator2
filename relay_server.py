import argparse
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from utils.api_uploads import (
    InvalidImageUpload,
    UploadTooLarge,
    copy_upload_file,
    max_upload_bytes_from_env,
    upload_too_large_message,
    validate_image_file,
    write_upload_bytes,
)


IMG_EXT = {".bmp", ".jpg", ".png", ".jpeg", ".webp", ".jxl"}
CONTENT_TYPE_EXT = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/jxl": ".jxl",
}
MEDIA_TYPES = {
    ".bmp": "image/bmp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jxl": "image/jxl",
}


@dataclass
class RelayJob:
    job_id: str
    job_dir: str
    input_path: str
    input_filename: str
    status: str
    created_at: float
    updated_at: float
    finished_at: Optional[float] = None
    worker_id: Optional[str] = None
    result_path: Optional[str] = None
    media_type: str = "image/png"
    error: Optional[str] = None
    status_code: int = 200

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "error": self.error,
            "status_code": self.status_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "input_filename": self.input_filename,
            "worker_id": self.worker_id,
            "status_url": f"/jobs/{self.job_id}",
            "result_url": f"/jobs/{self.job_id}/result",
        }

    def to_worker_dict(self) -> dict:
        data = self.to_dict()
        data["input_url"] = f"/worker/jobs/{self.job_id}/input"
        return data


class RelayJobStore:
    def __init__(self, storage_dir: str, result_ttl_seconds: int, max_upload_bytes: int):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.result_ttl_seconds = max(result_ttl_seconds, 1)
        self.max_upload_bytes = max(max_upload_bytes, 1)
        self._jobs = {}
        self._lock = threading.Lock()

    def submit(self, file: UploadFile, ext: str) -> RelayJob:
        self.cleanup()
        job_id = uuid.uuid4().hex
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        filename = file.filename or f"input{ext}"
        input_path = job_dir / f"input{ext}"

        try:
            copy_upload_file(file.file, input_path, self.max_upload_bytes)
            validate_image_file(input_path)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        now = time.time()
        job = RelayJob(
            job_id=job_id,
            job_dir=str(job_dir),
            input_path=str(input_path),
            input_filename=filename,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit_bytes(self, content: bytes, filename: str, ext: str) -> RelayJob:
        self.cleanup()
        job_id = uuid.uuid4().hex
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_filename = filename or f"input{ext}"
        input_path = job_dir / f"input{ext}"

        try:
            write_upload_bytes(content, input_path, self.max_upload_bytes)
            validate_image_file(input_path)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        now = time.time()
        job = RelayJob(
            job_id=job_id,
            job_dir=str(job_dir),
            input_path=str(input_path),
            input_filename=input_filename,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[RelayJob]:
        self.cleanup()
        with self._lock:
            return self._jobs.get(job_id)

    def wait_for_terminal(self, job_id: str, timeout_seconds: float, poll_interval: float) -> Optional[RelayJob]:
        deadline = time.time() + timeout_seconds
        poll_interval = max(poll_interval, 0.1)
        while time.time() < deadline:
            job = self.get(job_id)
            if job is None:
                return None
            if job.status in {"done", "failed"}:
                return job
            time.sleep(poll_interval)
        return self.get(job_id)

    def claim_next(self, worker_id: str) -> Optional[RelayJob]:
        self.cleanup()
        with self._lock:
            for job in self._jobs.values():
                if job.status == "queued":
                    job.status = "running"
                    job.worker_id = worker_id
                    job.updated_at = time.time()
                    return job
        return None

    def complete(self, job_id: str, result_path: str, media_type: str) -> Optional[RelayJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            now = time.time()
            job.status = "done"
            job.result_path = result_path
            job.media_type = media_type
            job.error = None
            job.status_code = 200
            job.updated_at = now
            job.finished_at = now
            return job

    def fail(self, job_id: str, error: str, status_code: int = 500) -> Optional[RelayJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            now = time.time()
            job.status = "failed"
            job.error = error
            job.status_code = status_code
            job.updated_at = now
            job.finished_at = now
            return job

    def stats(self) -> dict:
        with self._lock:
            counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            counts["total"] = len(self._jobs)
            return counts

    def cleanup(self) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.finished_at is not None and now - job.finished_at > self.result_ttl_seconds:
                    expired.append((job_id, job.job_dir))
            for job_id, _ in expired:
                self._jobs.pop(job_id, None)

        for _, job_dir in expired:
            shutil.rmtree(job_dir, ignore_errors=True)


def upload_extension(filename: str, content_type: str) -> Optional[str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMG_EXT:
        return suffix
    return CONTENT_TYPE_EXT.get(content_type)


def media_type_for_path(path: str) -> str:
    return MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def output_filename_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower() or ".png"
    return f"translated{suffix}"


def job_result_response(job: RelayJob) -> Response:
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=202, detail=job.to_dict())
    if job.status == "failed":
        raise HTTPException(status_code=job.status_code or 500, detail=job.error or "Translation failed.")
    if job.result_path is None or not Path(job.result_path).exists():
        raise HTTPException(status_code=410, detail="Result expired or missing.")
    return Response(
        content=Path(job.result_path).read_bytes(),
        media_type=job.media_type,
        headers={"Content-Disposition": f'attachment; filename="{output_filename_for_path(job.result_path)}"'},
    )


def make_auth_dependency(token: str):
    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    return require_auth


def raise_upload_http_exception(exc: Exception) -> None:
    if isinstance(exc, UploadTooLarge):
        raise HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidImageUpload):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=400, detail=str(exc))


def create_app(
    storage_dir: str,
    api_token: str = "",
    worker_token: str = "",
    result_ttl_seconds: int = 3600,
    max_upload_bytes: int = None,
):
    max_upload_bytes = max_upload_bytes or max_upload_bytes_from_env()
    store = RelayJobStore(storage_dir, result_ttl_seconds, max_upload_bytes)
    require_client_auth = make_auth_dependency(api_token)
    require_worker_auth = make_auth_dependency(worker_token or api_token)
    app = FastAPI(title="BalloonsTranslator Relay API")

    @app.middleware("http")
    async def reject_large_requests(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_upload_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": upload_too_large_message(max_upload_bytes)},
                        )
                except ValueError:
                    pass
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ok": True, "jobs": store.stats(), "max_upload_bytes": max_upload_bytes}

    @app.post("/translate")
    def translate_sync(
        file: UploadFile = File(...),
        timeout: float = Query(default=900.0, ge=1.0),
        poll_interval: float = Query(default=2.0, ge=0.1),
        _auth: None = Depends(require_client_auth),
    ):
        ext = upload_extension(file.filename, file.content_type)
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")
        try:
            job = store.submit(file, ext)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            raise_upload_http_exception(exc)

        completed = store.wait_for_terminal(job.job_id, timeout, poll_interval)
        if completed is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if completed.status in {"queued", "running"}:
            raise HTTPException(status_code=504, detail=completed.to_dict())
        return job_result_response(completed)

    @app.post("/translate/raw")
    def translate_raw_sync(
        content: bytes = Body(...),
        filename: str = Query(default="input.png"),
        timeout: float = Query(default=900.0, ge=1.0),
        poll_interval: float = Query(default=2.0, ge=0.1),
        content_type: Optional[str] = Header(default=None),
        _auth: None = Depends(require_client_auth),
    ):
        ext = upload_extension(filename, content_type or "")
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")
        try:
            job = store.submit_bytes(content, filename, ext)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            raise_upload_http_exception(exc)

        completed = store.wait_for_terminal(job.job_id, timeout, poll_interval)
        if completed is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if completed.status in {"queued", "running"}:
            raise HTTPException(status_code=504, detail=completed.to_dict())
        return job_result_response(completed)

    @app.post("/jobs", status_code=202)
    def create_job(file: UploadFile = File(...), _auth: None = Depends(require_client_auth)):
        ext = upload_extension(file.filename, file.content_type)
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")
        try:
            job = store.submit(file, ext)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            raise_upload_http_exception(exc)
        return job.to_dict()

    @app.post("/jobs/raw", status_code=202)
    def create_raw_job(
        content: bytes = Body(...),
        filename: str = Query(default="input.png"),
        content_type: Optional[str] = Header(default=None),
        _auth: None = Depends(require_client_auth),
    ):
        ext = upload_extension(filename, content_type or "")
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")
        try:
            job = store.submit_bytes(content, filename, ext)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            raise_upload_http_exception(exc)
        return job.to_dict()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, _auth: None = Depends(require_client_auth)):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job.to_dict()

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str, _auth: None = Depends(require_client_auth)):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job_result_response(job)

    @app.get("/worker/jobs/next")
    def claim_next_job(
        worker_id: str = Query(default="default"),
        _auth: None = Depends(require_worker_auth),
    ):
        job = store.claim_next(worker_id)
        if job is None:
            return Response(status_code=204)
        return job.to_worker_dict()

    @app.get("/worker/jobs/{job_id}/input")
    def get_worker_input(job_id: str, _auth: None = Depends(require_worker_auth)):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        input_path = Path(job.input_path)
        if not input_path.exists():
            raise HTTPException(status_code=410, detail="Input expired or missing.")
        return Response(
            content=input_path.read_bytes(),
            media_type=media_type_for_path(job.input_path),
            headers={"Content-Disposition": f'attachment; filename="{job.input_filename}"'},
        )

    @app.post("/worker/jobs/{job_id}/result")
    def submit_worker_result(
        job_id: str,
        file: UploadFile = File(...),
        _auth: None = Depends(require_worker_auth),
    ):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        ext = upload_extension(file.filename, file.content_type) or ".png"
        result_path = Path(job.job_dir) / f"result{ext}"
        try:
            copy_upload_file(file.file, result_path, max_upload_bytes)
            validate_image_file(result_path)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            result_path.unlink(missing_ok=True)
            raise_upload_http_exception(exc)
        completed = store.complete(job_id, str(result_path), file.content_type or media_type_for_path(str(result_path)))
        return completed.to_dict()

    @app.post("/worker/jobs/{job_id}/failed")
    def submit_worker_failure(
        job_id: str,
        payload: dict,
        _auth: None = Depends(require_worker_auth),
    ):
        status_code = int(payload.get("status_code") or 500)
        error = str(payload.get("error") or "Worker failed.")
        job = store.fail(job_id, error, status_code=status_code)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job.to_dict()

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="BalloonsTranslator public relay server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--storage-dir", default=os.environ.get("BALLOONTRANS_RELAY_STORAGE", "relay_storage"))
    parser.add_argument("--api-token", default=os.environ.get("BALLOONTRANS_RELAY_API_TOKEN", os.environ.get("BALLOONTRANS_API_TOKEN", "")))
    parser.add_argument("--worker-token", default=os.environ.get("BALLOONTRANS_RELAY_WORKER_TOKEN", ""))
    parser.add_argument("--result-ttl", default=3600, type=int)
    parser.add_argument("--max-upload-mb", default=50, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(
        args.storage_dir,
        args.api_token,
        args.worker_token,
        args.result_ttl,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
