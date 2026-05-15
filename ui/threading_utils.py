from typing import Iterable

from qtpy.QtCore import QThread

from utils.logger import logger as LOGGER


def stop_qthread(thread: QThread, timeout_ms: int = 3000, name: str = None) -> bool:
    if thread is None or not thread.isRunning():
        return True

    request_stop = getattr(thread, "requestStop", None)
    if callable(request_stop):
        request_stop()

    thread.quit()
    if thread.wait(timeout_ms):
        return True

    LOGGER.warning(f'{name or thread.__class__.__name__} did not stop within {timeout_ms} ms.')
    return False


def wait_if_running(thread: QThread) -> None:
    if thread is not None and thread.isRunning():
        thread.wait()


def any_thread_running(threads: Iterable[QThread]) -> bool:
    return any(thread is not None and thread.isRunning() for thread in threads)
