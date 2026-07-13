import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Optional

from qtpy.QtCore import QObject, Signal, QTimer

from ui.threading_utils import any_thread_running, wait_if_running
from utils.config import pcfg
from utils.io_utils import IMG_EXT
from utils.logger import logger as LOGGER
from utils.api_uploads import (
    InvalidImageUpload,
    UploadTooLarge,
    copy_upload_file,
    max_upload_bytes_from_env,
    upload_too_large_message,
    validate_image_file,
)
from utils.api_security import require_auth_for_public_bind


_CONTENT_TYPE_EXT = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/jxl": ".jxl",
}

_MEDIA_TYPES = {
    ".bmp": "image/bmp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jxl": "image/jxl",
}


@dataclass
class TranslationOutcome:
    ok: bool
    status_code: int = 200
    result_path: Optional[str] = None
    media_type: str = "image/png"
    error: str = ""


@dataclass
class ApiJobRecord:
    job_id: str
    project_dir: str
    input_filename: str
    status: str = "queued"
    created_at: float = 0
    updated_at: float = 0
    finished_at: Optional[float] = None
    result_path: Optional[str] = None
    media_type: str = "image/png"
    error: Optional[str] = None
    status_code: int = 200

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "input_filename": self.input_filename,
            "status_url": f"/jobs/{self.job_id}",
            "result_url": f"/jobs/{self.job_id}/result",
        }


class TranslationJob:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.outcomes = Queue(maxsize=1)

    def finish(self, outcome: TranslationOutcome) -> None:
        self.outcomes.put(outcome)


class HeadlessTranslationBridge(QObject):
    start_translation = Signal(object)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._current_job: Optional[TranslationJob] = None
        self._busy = False
        self._busy_lock = threading.Lock()
        self.start_translation.connect(self._start_translation)

    @property
    def busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    def translate(self, project_dir: str) -> TranslationOutcome:
        job = TranslationJob(project_dir)
        self.start_translation.emit(job)
        return job.outcomes.get()

    def _set_busy(self, busy: bool) -> None:
        with self._busy_lock:
            self._busy = busy

    def _module_threads_running(self) -> bool:
        manager = self.main_window.module_manager
        return any_thread_running([
            manager.textdetect_thread,
            manager.ocr_thread,
            manager.translate_thread,
            manager.inpaint_thread,
            manager.imgtrans_thread,
        ])

    def _missing_modules(self):
        manager = self.main_window.module_manager
        missing = []
        if pcfg.module.enable_detect and manager.textdetector is None:
            missing.append("text detector")
        if pcfg.module.enable_ocr and manager.ocr is None:
            missing.append("OCR")
        if pcfg.module.enable_translate and manager.translator is None:
            missing.append("translator")
        if pcfg.module.enable_inpaint and manager.inpainter is None:
            missing.append("inpainter")
        return missing

    def _start_translation(self, job: TranslationJob) -> None:
        if self._current_job is not None:
            job.finish(TranslationOutcome(False, 409, error="Another translation job is already running."))
            return

        if self._module_threads_running():
            QTimer.singleShot(100, lambda: self._start_translation(job))
            return

        missing_modules = self._missing_modules()
        if missing_modules:
            job.finish(
                TranslationOutcome(
                    False,
                    503,
                    error="Required modules are not ready: " + ", ".join(missing_modules),
                )
            )
            return

        self._current_job = job
        self._set_busy(True)
        try:
            self.main_window.module_manager.imgtrans_pipeline_finished.connect(self._finish_translation)
            self.main_window.openDir(job.project_dir)
            if self.main_window.imgtrans_proj.directory != job.project_dir:
                self._finish_current_job(TranslationOutcome(False, 400, error="Failed to load uploaded image."))
                return
            if self.main_window.imgtrans_proj.is_empty:
                self._finish_current_job(TranslationOutcome(False, 400, error="No supported image found."))
                return
            if not self.main_window.imgtrans_proj.img_valid:
                self._finish_current_job(TranslationOutcome(False, 400, error="Uploaded image could not be decoded."))
                return
            self.main_window.on_run_imgtrans()
        except Exception as exc:
            LOGGER.exception("Failed to start headless translation job.")
            self._finish_current_job(TranslationOutcome(False, 500, error=str(exc)))

    def _finish_translation(self) -> None:
        job = self._current_job
        if job is None:
            return

        try:
            wait_if_running(self.main_window.imsave_thread)
            page_name = self.main_window.imgtrans_proj.current_img
            if page_name is None and self.main_window.imgtrans_proj.pages:
                page_name = next(iter(self.main_window.imgtrans_proj.pages))
            if page_name is None:
                self._finish_current_job(TranslationOutcome(False, 500, error="No translated page found."))
                return

            result_path = self.main_window.imgtrans_proj.get_result_path(page_name)
            if not Path(result_path).exists():
                self._finish_current_job(
                    TranslationOutcome(False, 500, error=f"Result image was not written: {result_path}")
                )
                return

            media_type = media_type_for_path(result_path)
            self._finish_current_job(TranslationOutcome(True, result_path=result_path, media_type=media_type))
        except Exception as exc:
            LOGGER.exception("Failed to finish headless translation job.")
            self._finish_current_job(TranslationOutcome(False, 500, error=str(exc)))

    def _finish_current_job(self, outcome: TranslationOutcome) -> None:
        job = self._current_job
        self._current_job = None
        self._set_busy(False)
        try:
            self.main_window.module_manager.imgtrans_pipeline_finished.disconnect(self._finish_translation)
        except Exception:
            pass
        if job is not None:
            job.finish(outcome)


class TranslationJobManager:
    def __init__(self, bridge: HeadlessTranslationBridge, translation_lock: threading.Lock, result_ttl_seconds: int):
        self.bridge = bridge
        self.translation_lock = translation_lock
        self.result_ttl_seconds = max(result_ttl_seconds, 1)
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="headless-api-job-worker", daemon=True)
        self._worker.start()

    def submit(self, project_dir: str, input_filename: str) -> ApiJobRecord:
        self.cleanup()
        now = time.time()
        record = ApiJobRecord(
            job_id=uuid.uuid4().hex,
            project_dir=project_dir,
            input_filename=input_filename,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[record.job_id] = record
        self._queue.put(record.job_id)
        return record

    def get(self, job_id: str) -> Optional[ApiJobRecord]:
        self.cleanup()
        with self._lock:
            return self._jobs.get(job_id)

    def has_pending_jobs(self) -> bool:
        with self._lock:
            return any(job.status in {"queued", "running"} for job in self._jobs.values())

    def cleanup(self) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.finished_at is not None and now - job.finished_at > self.result_ttl_seconds:
                    expired.append((job_id, job.project_dir))
            for job_id, _ in expired:
                self._jobs.pop(job_id, None)

        for _, project_dir in expired:
            shutil.rmtree(project_dir, ignore_errors=True)

    def _set_status(self, job_id: str, status: str) -> Optional[ApiJobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = status
            job.updated_at = time.time()
            return job

    def _finish_job(self, job_id: str, outcome: TranslationOutcome) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = time.time()
            job.updated_at = now
            job.finished_at = now
            job.status_code = outcome.status_code
            if outcome.ok:
                job.status = "done"
                job.result_path = outcome.result_path
                job.media_type = outcome.media_type
                job.error = None
            else:
                job.status = "failed"
                job.error = outcome.error

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                job = self._set_status(job_id, "running")
                if job is None:
                    continue
                with self.translation_lock:
                    outcome = self.bridge.translate(job.project_dir)
                self._finish_job(job_id, outcome)
            except Exception as exc:
                LOGGER.exception("Headless API job failed.")
                self._finish_job(job_id, TranslationOutcome(False, 500, error=str(exc)))
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=2.0)


def media_type_for_path(path: str) -> str:
    return _MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def output_filename_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower() or ".png"
    return f"translated{suffix}"


def upload_extension(filename: str, content_type: str) -> Optional[str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMG_EXT:
        return suffix
    return _CONTENT_TYPE_EXT.get(content_type)


def save_upload_to_project(file, ext: str, project_dir: str, max_upload_bytes: int) -> Path:
    filename = f"input{ext}"
    input_path = Path(project_dir) / filename
    copy_upload_file(file.file, input_path, max_upload_bytes)
    validate_image_file(input_path)
    return input_path


def raise_upload_http_exception(exc: Exception) -> None:
    from fastapi import HTTPException

    if isinstance(exc, UploadTooLarge):
        raise HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidImageUpload):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=400, detail=str(exc))


def create_app(
    bridge: HeadlessTranslationBridge,
    translation_lock: threading.Lock,
    api_token: str = "",
    result_ttl_seconds: int = 3600,
    max_upload_bytes: int = None,
):
    from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse, Response

    max_upload_bytes = max_upload_bytes or max_upload_bytes_from_env()
    job_manager = TranslationJobManager(bridge, translation_lock, result_ttl_seconds)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            job_manager.close()

    app = FastAPI(title="BalloonsTranslator Headless API", lifespan=lifespan)
    app.state.job_manager = job_manager

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

    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not api_token:
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "busy": translation_lock.locked() or bridge.busy,
            "queued": job_manager.has_pending_jobs(),
            "result_format": pcfg.imgsave_ext,
            "max_upload_bytes": max_upload_bytes,
        }

    @app.post("/translate")
    def translate(file: UploadFile = File(...), _auth: None = Depends(require_auth)):
        ext = upload_extension(file.filename, file.content_type)
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        with translation_lock:
            with tempfile.TemporaryDirectory(prefix="balloontrans-api-") as tmp_dir:
                try:
                    save_upload_to_project(file, ext, tmp_dir, max_upload_bytes)
                except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
                    raise_upload_http_exception(exc)

                outcome = bridge.translate(tmp_dir)
                if not outcome.ok:
                    raise HTTPException(status_code=outcome.status_code, detail=outcome.error)

                result_path = Path(outcome.result_path)
                content = result_path.read_bytes()
                return Response(
                    content=content,
                    media_type=outcome.media_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{output_filename_for_path(outcome.result_path)}"'
                    },
                )

    @app.post("/jobs", status_code=202)
    def create_job(file: UploadFile = File(...), _auth: None = Depends(require_auth)):
        ext = upload_extension(file.filename, file.content_type)
        if ext is None:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        project_dir = tempfile.mkdtemp(prefix="balloontrans-api-job-")
        try:
            save_upload_to_project(file, ext, project_dir, max_upload_bytes)
        except (ValueError, UploadTooLarge, InvalidImageUpload) as exc:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise_upload_http_exception(exc)
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise

        job = job_manager.submit(project_dir, file.filename or f"input{ext}")
        return job.to_dict()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, _auth: None = Depends(require_auth)):
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job.to_dict()

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str, _auth: None = Depends(require_auth)):
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status in {"queued", "running"}:
            raise HTTPException(status_code=202, detail=job.to_dict())
        if job.status == "failed":
            raise HTTPException(status_code=job.status_code or 500, detail=job.error or "Translation failed.")
        if job.result_path is None or not Path(job.result_path).exists():
            raise HTTPException(status_code=410, detail="Result expired or missing.")

        return Response(
            content=Path(job.result_path).read_bytes(),
            media_type=job.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename_for_path(job.result_path)}"'
            },
        )

    return app


def start_headless_server(
    main_window,
    host: str,
    port: int,
    api_token: str = "",
    result_ttl_seconds: int = 3600,
    max_upload_bytes: int = None,
    allow_unauthenticated_public: bool = False,
):
    import uvicorn

    require_auth_for_public_bind(
        host,
        {'api': api_token},
        allow_unauthenticated_public=allow_unauthenticated_public,
    )
    bridge = HeadlessTranslationBridge(main_window)
    translation_lock = threading.Lock()
    app = create_app(bridge, translation_lock, api_token, result_ttl_seconds, max_upload_bytes)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="headless-http-server", daemon=True)
    thread.start()
    LOGGER.info(f"Headless API server listening on http://{host}:{port}")
    return server, thread
