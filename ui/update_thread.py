from typing import Optional

from qtpy.QtCore import QThread, Signal

from utils.updater import BallonsTranslatorUpdater, ReleaseInfo


class UpdateCheckThread(QThread):
    update_progress = Signal(dict)
    finish_check = Signal(object)
    failed = Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = 'check'
        self.release_info: Optional[ReleaseInfo] = None
        self.current_version = None

    def checkUpdate(self) -> bool:
        if self.isRunning():
            return False
        self.mode = 'check'
        self.release_info = None
        self.current_version = None
        self.start()
        return True

    def applyUpdate(self, release_info: ReleaseInfo, current_version: str = None) -> bool:
        if self.isRunning():
            return False
        self.mode = 'apply'
        self.release_info = release_info
        self.current_version = current_version
        self.start()
        return True

    def _emit_progress(self, payload: dict):
        self.update_progress.emit(dict(payload))

    def run(self):
        updater = BallonsTranslatorUpdater(progress_callback=self._emit_progress)
        try:
            if self.mode == 'apply':
                result = updater.apply_update(self.release_info, self.current_version)
            else:
                result = updater.check_latest_release()
            self.finish_check.emit(result)
        except Exception as e:
            self.failed.emit(e)
