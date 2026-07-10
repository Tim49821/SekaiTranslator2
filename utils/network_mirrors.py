import json
import locale
import os
import time
from typing import Iterable, Optional, Set
from urllib.request import getproxies


HUGGINGFACE_ORIGIN = 'https://huggingface.co'
DEFAULT_HUGGINGFACE_MIRROR = 'https://hf-mirror.com'
DEFAULT_PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple'
HUGGINGFACE_MIRROR_OPTIONS = (None, DEFAULT_HUGGINGFACE_MIRROR)
PYPI_MIRROR_OPTIONS = (None, DEFAULT_PYPI_MIRROR)
MIRROR_FIELDS = ('huggingface', 'pypi')


def normalize_mirror_value(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() == 'none':
        return None
    return value.rstrip('/')


def mirror_to_display(value: Optional[str], none_label: str = 'None') -> str:
    return none_label if normalize_mirror_value(value) is None else normalize_mirror_value(value)


def mirror_from_display(value: str, none_label: str = 'None') -> Optional[str]:
    if value == none_label:
        return None
    return normalize_mirror_value(value)


def display_options(values: Iterable[Optional[str]], none_label: str = 'None') -> list:
    return [mirror_to_display(value, none_label=none_label) for value in values]


def rewrite_huggingface_url(url: str, mirror: Optional[str]) -> str:
    mirror = normalize_mirror_value(mirror)
    if not mirror or not isinstance(url, str):
        return url
    if url == HUGGINGFACE_ORIGIN:
        return mirror
    if url.startswith(HUGGINGFACE_ORIGIN + '/'):
        return mirror + url[len(HUGGINGFACE_ORIGIN):]
    return url


def installer_env_with_pypi_mirror(env: Optional[dict] = None, mirror: Optional[str] = None) -> dict:
    result = dict(env or os.environ.copy())
    mirror = normalize_mirror_value(mirror)
    if mirror:
        result['INDEX_URL'] = mirror
    return result


def read_saved_pypi_mirror(config_path: str) -> Optional[str]:
    mirrors = _read_raw_mirrors(config_path)
    if not isinstance(mirrors, dict):
        return None
    return normalize_mirror_value(mirrors.get('pypi'))


def missing_mirror_fields(config_path: str) -> Set[str]:
    data = _read_raw_config(config_path)
    if not isinstance(data, dict):
        return set(MIRROR_FIELDS)
    mirrors = data.get('mirrors')
    if not isinstance(mirrors, dict):
        return set(MIRROR_FIELDS)
    return {field for field in MIRROR_FIELDS if field not in mirrors}


def should_use_china_mirrors(locale_names: Iterable[str] = (), timezone_names: Iterable[str] = ()) -> bool:
    return _has_mainland_china_locale(locale_names) or _has_mainland_china_timezone(timezone_names)


def collect_system_locale_names() -> list:
    candidates = [
        os.environ.get('LC_ALL', ''),
        os.environ.get('LC_MESSAGES', ''),
        os.environ.get('LANG', ''),
    ]
    try:
        candidates.append(locale.getlocale()[0] or '')
    except Exception:
        pass
    try:
        candidates.append(locale.getdefaultlocale()[0] or '')
    except Exception:
        pass
    return _unique_nonempty(candidates)


def collect_system_timezone_names() -> list:
    candidates = [os.environ.get('TZ', '')]
    candidates.extend(name for name in time.tzname if name)
    candidates.append(_read_etc_timezone())
    candidates.append(_localtime_zoneinfo_name())
    return _unique_nonempty(candidates)


def backfill_missing_mirror_defaults(mirrors_config, missing_fields: Iterable[str], locale_names: Iterable[str] = (), timezone_names: Iterable[str] = ()) -> list:
    missing_fields = set(missing_fields)
    if not missing_fields or not should_use_china_mirrors(locale_names, timezone_names):
        return []
    updated = []
    if 'huggingface' in missing_fields and getattr(mirrors_config, 'huggingface', None) is None:
        mirrors_config.huggingface = DEFAULT_HUGGINGFACE_MIRROR
        updated.append('huggingface')
    if 'pypi' in missing_fields and getattr(mirrors_config, 'pypi', None) is None:
        mirrors_config.pypi = DEFAULT_PYPI_MIRROR
        updated.append('pypi')
    return updated


def has_effective_system_proxy() -> bool:
    proxies = getproxies()
    return any(key.lower() != 'no' and value for key, value in proxies.items())


def write_raw_mirror_config(config_path: str, huggingface: Optional[str], pypi: Optional[str]) -> bool:
    if not config_path:
        return False
    try:
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        tmp_save_tgt = config_path + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            json.dump({
                'mirrors': {
                    'huggingface': normalize_mirror_value(huggingface),
                    'pypi': normalize_mirror_value(pypi),
                }
            }, f, ensure_ascii=False)
        os.replace(tmp_save_tgt, config_path)
    except Exception:
        return False
    return True


def auto_fill_network_mirrors(config_path: str, logger=None) -> list:
    if config_path and os.path.exists(config_path):
        return []
    if has_effective_system_proxy():
        return []
    use_china_mirrors = should_use_china_mirrors(collect_system_locale_names(), collect_system_timezone_names())
    huggingface_mirror = DEFAULT_HUGGINGFACE_MIRROR if use_china_mirrors else None
    pypi_mirror = DEFAULT_PYPI_MIRROR if use_china_mirrors else None
    if not write_raw_mirror_config(config_path, huggingface_mirror, pypi_mirror):
        return []
    return list(MIRROR_FIELDS) if use_china_mirrors else []


def _has_mainland_china_locale(locale_names: Iterable[str]) -> bool:
    for value in locale_names:
        if not value:
            continue
        normalized = str(value).strip().split('.', 1)[0].replace('-', '_')
        lower = normalized.lower()
        upper = normalized.upper()
        if upper == 'CN' or lower == 'zh_cn' or lower.endswith('_cn') or '_cn_' in lower:
            return True
    return False


def _has_mainland_china_timezone(timezone_names: Iterable[str]) -> bool:
    for value in timezone_names:
        if not value:
            continue
        normalized = str(value).strip().lower().replace('\\', '/')
        if normalized in {'asia/shanghai', 'prc'}:
            return True
        if normalized.endswith('/asia/shanghai') or 'zoneinfo/asia/shanghai' in normalized:
            return True
        if 'china standard time' in normalized or '中国标准时间' in normalized or '中国夏令时' in normalized:
            return True
    return False


def _read_raw_config(config_path: str):
    if not config_path or not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return None


def _read_raw_mirrors(config_path: str):
    data = _read_raw_config(config_path)
    if not isinstance(data, dict):
        return None
    return data.get('mirrors')


def _read_etc_timezone() -> str:
    try:
        with open('/etc/timezone', 'r', encoding='utf8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _localtime_zoneinfo_name() -> str:
    try:
        path = os.path.realpath('/etc/localtime')
    except Exception:
        return ''
    marker = 'zoneinfo/'
    return path.split(marker, 1)[1] if marker in path else path


def _unique_nonempty(values: Iterable[str]) -> list:
    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique
