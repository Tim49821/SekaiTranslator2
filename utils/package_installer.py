import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from utils.logger import logger as LOGGER


BACKENDS = ('auto', 'pip', 'uv', 'conda-pip')
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
RAW_PROGRESS_RE = re.compile(r'^Progress\s+(\d+)\s+of\s+(\d+)$')
_PIP_RAW_PROGRESS_SUPPORT = {}


@dataclass
class InstallResult:
    ok: bool
    command: List[str]
    returncode: int = 0
    stdout: str = ''
    stderr: str = ''
    error: str = ''

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)


@dataclass
class _InstallerProgressState:
    download_message: str = ''
    total: int = 0
    last_downloaded: int = 0
    last_time: float = 0.0
    speed: float = 0.0


def resolve_backend(backend: str = 'auto', env: Optional[dict] = None, python_executable: str = '') -> str:
    if backend != 'auto':
        return backend if backend in BACKENDS else 'auto'
    if _find_uv_executable(env, python_executable):
        return 'uv'
    return 'pip'


def _find_uv_executable(env: Optional[dict] = None, python_executable: str = '') -> str:
    env = env or os.environ
    found = shutil.which('uv', path=env.get('PATH'))
    if found:
        return found
    executable_dir = Path(python_executable or sys.executable).parent
    for filename in ('uv.exe', 'uv.cmd', 'uv.bat', 'uv'):
        candidate = executable_dir / filename
        if candidate.is_file():
            return str(candidate)
    return ''


def build_install_command(
    requirements: Iterable[str] = (),
    requirements_file: str = '',
    constraint_files: Iterable[str] = (),
    backend: str = 'auto',
    extra_args: str = '',
    env: Optional[dict] = None,
    python_executable: str = '',
    python_prefix: str = '',
) -> List[str]:
    reqs = [req for req in dict.fromkeys(requirements) if req]
    if requirements_file:
        reqs.extend(['-r', requirements_file])
    constraints = [path for path in dict.fromkeys(constraint_files) if path]
    constraint_args = [arg for path in constraints for arg in ('-c', path)]
    extra = shlex.split(extra_args or '')
    env = env or os.environ
    index_args = ['-i', env['INDEX_URL']] if env.get('INDEX_URL') else []
    find_links_args = ['-f', env['FIND_LINKS']] if env.get('FIND_LINKS') else []
    python_executable = python_executable or sys.executable
    python_prefix = python_prefix or sys.prefix
    resolved_backend = resolve_backend(backend, env=env, python_executable=python_executable)

    if resolved_backend == 'uv':
        uv_executable = _find_uv_executable(env, python_executable) or 'uv'
        return [
            uv_executable, 'pip', 'install', '--python', python_executable,
            *reqs, *constraint_args, *find_links_args, *index_args, *extra,
        ]
    progress_args = _pip_progress_args(extra, env, python_executable)
    if resolved_backend == 'conda-pip':
        return [
            'conda', 'run', '-p', python_prefix,
            python_executable, '-m', 'pip', 'install',
            *reqs, *constraint_args, *progress_args, *find_links_args, *index_args, *extra,
        ]
    return [
        python_executable, '-m', 'pip', 'install',
        *reqs, *constraint_args,
        '--prefer-binary',
        '--disable-pip-version-check',
        '--no-warn-script-location',
        *progress_args,
        *find_links_args,
        *index_args,
        *extra,
    ]


def _pip_progress_args(extra: List[str], env: dict, python_executable: str = '') -> List[str]:
    if env.get('PIP_PROGRESS_BAR'):
        return []
    if any(arg == '--progress-bar' or arg.startswith('--progress-bar=') for arg in extra):
        return []
    if _pip_supports_raw_progress(python_executable or sys.executable, env):
        return ['--progress-bar', 'raw']
    return ['--progress-bar', 'off']


def _pip_supports_raw_progress(python_executable: str, env: dict) -> bool:
    if not python_executable:
        return False
    key = (python_executable, env.get('PATH', ''), env.get('PYTHONPATH', ''))
    if key in _PIP_RAW_PROGRESS_SUPPORT:
        return _PIP_RAW_PROGRESS_SUPPORT[key]
    probe_env = os.environ.copy()
    probe_env.update(env or {})
    try:
        completed = subprocess.run(
            [python_executable, '-m', 'pip', 'install', '--help'],
            env=probe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=False,
            timeout=10,
        )
    except Exception:
        _PIP_RAW_PROGRESS_SUPPORT[key] = False
        return False
    output = completed.stdout or ''
    supports_raw = completed.returncode == 0 and '--progress-bar' in output and 'raw' in output
    _PIP_RAW_PROGRESS_SUPPORT[key] = supports_raw
    return supports_raw


def install(
    requirements: Iterable[str] = (),
    requirements_file: str = '',
    constraint_files: Iterable[str] = (),
    backend: str = 'auto',
    extra_args: str = '',
    env: Optional[dict] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> InstallResult:
    install_env = env or os.environ.copy()
    command = build_install_command(
        requirements=requirements,
        requirements_file=requirements_file,
        constraint_files=constraint_files,
        backend=backend,
        extra_args=extra_args,
        env=install_env,
    )
    LOGGER.info(f'Using Python package installer backend: {resolve_backend(backend, env=install_env)}')
    if install_env.get('INDEX_URL'):
        LOGGER.info(f'Using PyPI package mirror for package install: {install_env["INDEX_URL"]}')
    if _can_stream_with_pty():
        try:
            returncode, output = _run_with_pty(command, env=install_env, progress_callback=progress_callback)
        except Exception as e:
            return InstallResult(False, command, error=str(e), returncode=-1)
    else:
        try:
            process = subprocess.Popen(
                command,
                env=install_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=False,
                bufsize=1,
            )
        except Exception as e:
            return InstallResult(False, command, error=str(e), returncode=-1)
        output = _stream_process_output(process, progress_callback=progress_callback)
        returncode = process.wait()
    return InstallResult(returncode == 0, command, returncode=returncode, stdout=output)


def _can_stream_with_pty() -> bool:
    if os.name == 'nt':
        return False
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    try:
        import pty  # noqa: F401
    except Exception:
        return False
    return True


def _run_with_pty(
    command: List[str],
    env: Optional[dict] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> Tuple[int, str]:
    import pty

    master_fd, slave_fd = pty.openpty()
    try:
        process = subprocess.Popen(
            command,
            env=env or os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=False,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    captured = []
    pending = []
    progress_state = _InstallerProgressState()
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode(errors='replace')
                captured.append(text)
                _feed_progress_text(text, pending, progress_callback, progress_state, echo=True)
            elif process.poll() is not None:
                break
    finally:
        os.close(master_fd)
    _emit_progress_message(pending, progress_callback, progress_state, echo=True)
    return process.wait(), ''.join(captured)


def _stream_process_output(process: subprocess.Popen, progress_callback: Optional[Callable[[dict], None]] = None) -> str:
    captured = []
    pending = []
    progress_state = _InstallerProgressState()
    while True:
        chunk = process.stdout.read(1) if process.stdout is not None else ''
        if chunk == '':
            if process.poll() is not None:
                break
            continue
        captured.append(chunk)
        _feed_progress_text(chunk, pending, progress_callback, progress_state, echo=True)
    _emit_progress_message(pending, progress_callback, progress_state, echo=True)
    return ''.join(captured)


def _feed_progress_text(text: str, pending: List[str], progress_callback=None, progress_state=None, echo=False):
    for char in text:
        if char in {'\n', '\r'}:
            _emit_progress_message(pending, progress_callback, progress_state, echo=echo)
        else:
            pending.append(char)
            if len(pending) >= 200:
                _emit_progress_message(pending, progress_callback, progress_state, echo=echo)


def _print_stream_text(text: str):
    try:
        print(text, end='', flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        safe_text = text.encode(encoding, errors='replace').decode(encoding, errors='replace')
        print(safe_text, end='', flush=True)


def _emit_progress_message(pending: List[str], progress_callback=None, progress_state=None, echo=False):
    if not pending:
        return
    message = ANSI_ESCAPE_RE.sub('', ''.join(pending)).strip()
    pending.clear()
    if not message:
        return
    progress_state = progress_state or _InstallerProgressState()
    payload = _package_download_progress_payload(message, progress_state)
    if payload is None:
        payload = {'event': 'package_output', 'message': message}
    if echo:
        if payload.get('event') == 'package_download_progress':
            _print_stream_text(_format_download_progress_line(payload) + '\n')
        else:
            _print_stream_text(payload.get('message', message) + '\n')
    if progress_callback is not None:
        progress_callback(payload)


def _package_download_progress_payload(message: str, state: _InstallerProgressState) -> Optional[dict]:
    if message.startswith('Downloading '):
        state.download_message = _download_display_message(message)
        return {'event': 'package_output', 'message': state.download_message}
    match = RAW_PROGRESS_RE.match(message)
    if match is None:
        return None
    downloaded = int(match.group(1))
    total = int(match.group(2))
    now = time.monotonic()
    if downloaded < state.last_downloaded or total != state.total or not state.last_time:
        state.speed = 0.0
    else:
        elapsed = max(now - state.last_time, 1e-6)
        delta = max(downloaded - state.last_downloaded, 0)
        instant_speed = delta / elapsed
        if instant_speed > 0:
            state.speed = instant_speed if not state.speed else (state.speed * 0.7 + instant_speed * 0.3)
    state.last_time = now
    state.total = total
    state.last_downloaded = downloaded
    eta = None
    if total and state.speed > 0 and downloaded < total:
        eta = max(int(round((total - downloaded) / state.speed)), 0)
    return {
        'event': 'package_download_progress',
        'message': state.download_message or 'Downloading package',
        'downloaded': downloaded,
        'total': total or None,
        'speed': state.speed or None,
        'eta': eta,
    }


def _format_download_progress_line(payload: dict) -> str:
    downloaded = payload.get('downloaded') or 0
    total = payload.get('total')
    parts = [payload.get('message') or 'Downloading package']
    parts.append(f'{downloaded / total * 100:.1f}%' if total else _sizeof_fmt(downloaded))
    if payload.get('speed'):
        parts.append(f'{_sizeof_fmt(payload["speed"])}/s')
    if payload.get('eta') is not None:
        parts.append(f'ETA {_format_duration(payload["eta"])}')
    return ' | '.join(parts)


def _download_display_message(message: str) -> str:
    target = message[len('Downloading '):].strip()
    target = re.sub(r'\s+\([^)]*\)\s*$', '', target)
    parsed = urlparse(target)
    if parsed.scheme and parsed.path:
        target = unquote(parsed.path.rsplit('/', 1)[-1])
    return f'Downloading {_simple_package_name_from_download_target(target)}'


def _simple_package_name_from_download_target(target: str) -> str:
    name = re.sub(r'\.metadata$', '', (target or '').strip())
    name = re.sub(r'(\.tar\.gz|\.zip|\.whl|\.tgz|\.tar\.bz2)$', '', name, flags=re.IGNORECASE)
    match = re.match(r'(.+?)-(?=\d)', name)
    return match.group(1) if match else (name or 'package')


def _sizeof_fmt(size, suffix='B') -> str:
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'


def _format_duration(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'
