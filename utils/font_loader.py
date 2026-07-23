import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import List


FONT_EXTENSIONS = {'.ttf', '.otf', '.ttc', '.pfb'}


def unique_font_families(families) -> List[str]:
    unique = {}
    for family in families:
        family = str(family).strip()
        if family:
            unique.setdefault(family.casefold(), family)
    return sorted(unique.values(), key=str.casefold)


def resolve_custom_font_family(family: str, families=None) -> str:
    if families is None:
        from . import shared

        families = shared.CUSTOM_FONTS

    families = unique_font_families(families)
    if not families:
        return ''

    requested = str(family or '').strip().casefold()
    for available_family in families:
        if available_family.casefold() == requested:
            return available_family

    for preferred_family in ('Pretendard Variable', 'Pretendard'):
        for available_family in families:
            if available_family.casefold() == preferred_family.casefold():
                return available_family
    return families[0]


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


def load_custom_font_families(font_paths) -> List[str]:
    from qtpy.QtGui import QFontDatabase

    families = []
    for font_path in font_paths:
        font_id = add_application_font(str(font_path))
        if font_id >= 0:
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
    return unique_font_families(families)
