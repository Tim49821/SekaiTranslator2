import hashlib
import shutil
import tempfile
from pathlib import Path


FONT_EXTENSIONS = {'.ttf', '.otf', '.ttc', '.pfb'}


def _clean_font_copy_path(font_path: Path) -> Path:
    digest = hashlib.sha1(str(font_path.resolve()).encode('utf8')).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / 'seka_translator_fonts' / f'{font_path.stem}_{digest}{font_path.suffix}'


def add_application_font(font_path: str) -> int:
    from qtpy.QtGui import QFontDatabase

    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id >= 0:
        return font_id

    src_path = Path(font_path)
    try:
        clean_path = _clean_font_copy_path(src_path)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        if not clean_path.exists() or clean_path.stat().st_size != src_path.stat().st_size:
            shutil.copyfile(src_path, clean_path)
        return QFontDatabase.addApplicationFont(str(clean_path))
    except Exception:
        return font_id
