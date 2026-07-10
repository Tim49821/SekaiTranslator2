import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from utils import shared


TORCH_FAMILY_PACKAGES = {'torch', 'torchvision', 'torchaudio'}
TORCH_INSTALL_DEVICE_OPTIONS = ('cpu', 'cuda', 'xpu')
TORCH_CUDA_VERSION_OPTIONS = ('cu128', 'cu118')
TORCH_CUDA_CUTOFF = 7.5
NVIDIA_SMI_TIMEOUT = 5
XPU_SMI_TIMEOUT = 5
ALIYUN_PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple'
ALIYUN_PYTORCH_WHEEL_ROOT = 'https://mirrors.aliyun.com/pytorch-wheels'


@dataclass(frozen=True)
class NvidiaGpuInfo:
    name: str
    compute_capability: Optional[float] = None


@dataclass(frozen=True)
class IntelXpuInfo:
    name: str
    device_id: Optional[str] = None


@dataclass(frozen=True)
class TorchInstallProfile:
    name: str
    requirements: Sequence[str]
    index_url: str
    use_aliyun_find_links: bool = True


@dataclass(frozen=True)
class TorchInstallRequest:
    requirements: List[str]
    env: dict
    profile: Optional[TorchInstallProfile] = None
    backend: Optional[str] = None
    device: str = 'cpu'
    cuda_version: Optional[str] = None


OLDER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu118',
    requirements=('torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1'),
    index_url='https://download.pytorch.org/whl/cu118',
)
NEWER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu128',
    requirements=('torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0'),
    index_url='https://download.pytorch.org/whl/cu128',
)
INTEL_XPU_PROFILE = TorchInstallProfile(
    name='xpu',
    requirements=('torch', 'torchvision', 'torchaudio'),
    index_url='https://download.pytorch.org/whl/xpu',
    use_aliyun_find_links=False,
)


def prepare_torch_install_request(
    requirements: Iterable[str],
    env: Optional[dict] = None,
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
    torch_device: Optional[str] = None,
    torch_cuda_version: Optional[str] = None,
) -> TorchInstallRequest:
    reqs = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
    request_env = dict(env or os.environ.copy())
    if not _has_plain_unpinned_torch(reqs):
        return TorchInstallRequest(reqs, request_env)

    profile, device = select_torch_install_profile_for_device(
        torch_device,
        torch_cuda_version=torch_cuda_version,
        gpu_detector=gpu_detector,
        xpu_detector=xpu_detector,
    )
    if profile is None:
        return TorchInstallRequest(reqs, request_env, device=device)

    return TorchInstallRequest(
        requirements=_rewrite_torch_family_requirements(reqs, profile),
        env=_env_for_torch_profile(request_env, profile),
        profile=profile,
        backend='pip',
        device=device,
        cuda_version=profile.name if device == 'cuda' else None,
    )


def _env_for_torch_profile(env: dict, profile: TorchInstallProfile) -> dict:
    result = dict(env)
    if profile.use_aliyun_find_links and _is_aliyun_pypi_mirror(result.get('INDEX_URL')):
        result['INDEX_URL'] = ALIYUN_PYPI_MIRROR
        result['FIND_LINKS'] = f'{ALIYUN_PYTORCH_WHEEL_ROOT}/{profile.name}'
    else:
        result['INDEX_URL'] = profile.index_url
        result.pop('FIND_LINKS', None)
    return result


def select_torch_install_profile(gpus: Sequence[NvidiaGpuInfo]) -> Optional[TorchInstallProfile]:
    if not gpus:
        return None
    for gpu in gpus:
        if gpu.compute_capability is None or gpu.compute_capability < TORCH_CUDA_CUTOFF:
            return OLDER_NVIDIA_PROFILE
    return NEWER_NVIDIA_PROFILE


def select_torch_install_profile_for_device(
    torch_device: Optional[str] = None,
    torch_cuda_version: Optional[str] = None,
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    if torch_device is not None:
        torch_device = torch_device.lower()
    if torch_device not in (None, *TORCH_INSTALL_DEVICE_OPTIONS):
        torch_device = None
    profile_by_cuda_version = {
        OLDER_NVIDIA_PROFILE.name: OLDER_NVIDIA_PROFILE,
        NEWER_NVIDIA_PROFILE.name: NEWER_NVIDIA_PROFILE,
    }
    if torch_device == 'cpu':
        return None, 'cpu'
    if torch_device == 'cuda':
        if torch_cuda_version in profile_by_cuda_version:
            return profile_by_cuda_version[torch_cuda_version], 'cuda'
        detector = gpu_detector or detect_nvidia_gpus
        return select_torch_install_profile(detector()) or NEWER_NVIDIA_PROFILE, 'cuda'
    if torch_device == 'xpu':
        return INTEL_XPU_PROFILE, 'xpu'
    return _cached_preferred_torch_install_profile(gpu_detector, xpu_detector)


def _cached_preferred_torch_install_profile(
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    if gpu_detector is not None or xpu_detector is not None:
        return _detect_preferred_torch_install_profile(gpu_detector, xpu_detector)

    cached_device = getattr(shared, 'TORCH_INSTALL_PREFERRED_DEVICE', None)
    if cached_device in TORCH_INSTALL_DEVICE_OPTIONS:
        return getattr(shared, 'TORCH_INSTALL_PREFERRED_PROFILE', None), cached_device

    profile, device = _detect_preferred_torch_install_profile()
    shared.TORCH_INSTALL_PREFERRED_DEVICE = device
    shared.TORCH_INSTALL_PREFERRED_PROFILE = profile
    return profile, device


def _detect_preferred_torch_install_profile(
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    profile = select_torch_install_profile((gpu_detector or detect_nvidia_gpus)())
    if profile is not None:
        return profile, 'cuda'
    xpu_profile = select_torch_xpu_install_profile((xpu_detector or detect_intel_xpus)())
    if xpu_profile is not None:
        return xpu_profile, 'xpu'
    return None, 'cpu'


def select_torch_xpu_install_profile(xpus: Sequence[IntelXpuInfo]) -> Optional[TorchInstallProfile]:
    return INTEL_XPU_PROFILE if xpus else None


def detect_nvidia_gpus() -> List[NvidiaGpuInfo]:
    if sys.platform not in {'win32', 'linux'}:
        return []
    command_path = _find_nvidia_smi()
    if not command_path:
        return []
    output = _run_command([
        command_path,
        '--query-gpu=name,compute_cap',
        '--format=csv,noheader,nounits',
    ], NVIDIA_SMI_TIMEOUT)
    gpus = _parse_nvidia_smi_compute_output(output)
    if gpus:
        return gpus
    name_output = _run_command([
        command_path,
        '--query-gpu=name',
        '--format=csv,noheader,nounits',
    ], NVIDIA_SMI_TIMEOUT)
    return [NvidiaGpuInfo(name.strip(), None) for name in name_output.splitlines() if name.strip()]


def detect_intel_xpus() -> List[IntelXpuInfo]:
    if sys.platform not in {'win32', 'linux'}:
        return []
    command_path = _find_xpu_smi()
    if not command_path:
        return []
    output = _run_command([command_path, 'discovery', '-j'], XPU_SMI_TIMEOUT)
    xpus = _parse_xpu_smi_discovery_json(output)
    if xpus:
        return xpus
    return _parse_xpu_smi_discovery_text(_run_command([command_path, 'discovery'], XPU_SMI_TIMEOUT))


def _find_nvidia_smi() -> Optional[str]:
    found = shutil.which('nvidia-smi')
    if found:
        return found
    candidates = []
    if sys.platform == 'win32':
        system_root = os.environ.get('SystemRoot')
        program_files = os.environ.get('ProgramFiles')
        if system_root:
            candidates.append(os.path.join(system_root, 'System32', 'nvidia-smi.exe'))
        if program_files:
            candidates.append(os.path.join(program_files, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'))
    elif sys.platform == 'linux':
        candidates.extend(['/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi'])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _find_xpu_smi() -> Optional[str]:
    found = shutil.which('xpu-smi')
    if found:
        return found
    candidates = []
    if sys.platform == 'win32':
        for program_dir in (os.environ.get('ProgramFiles'), os.environ.get('ProgramFiles(x86)')):
            if program_dir:
                candidates.extend([
                    os.path.join(program_dir, 'Intel', 'oneAPI', 'tools', 'latest', 'xpu-smi', 'xpu-smi.exe'),
                    os.path.join(program_dir, 'Intel', 'oneAPI', 'tools', 'latest', 'bin', 'xpu-smi.exe'),
                ])
    elif sys.platform == 'linux':
        candidates.extend(['/usr/bin/xpu-smi', '/usr/local/bin/xpu-smi', '/opt/intel/xpumanager/bin/xpu-smi'])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _run_command(command: Sequence[str], timeout: int) -> str:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=timeout,
        )
    except Exception:
        return ''
    return completed.stdout if completed.returncode == 0 else ''


def _parse_nvidia_smi_compute_output(output: str) -> List[NvidiaGpuInfo]:
    gpus = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, capability_text = line.partition(',')
        name = name.strip()
        if name:
            gpus.append(NvidiaGpuInfo(name, _parse_compute_capability(capability_text)))
    return gpus


def _parse_xpu_smi_discovery_json(output: str) -> List[IntelXpuInfo]:
    if not output:
        return []
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        devices = data.get('device_list') or data.get('devices') or []
    elif isinstance(data, list):
        devices = data
    else:
        return []
    xpus = []
    for entry in devices:
        if not isinstance(entry, dict):
            continue
        name = _first_entry_value(entry, ('device_name', 'name', 'Device Name'))
        device_id = _first_entry_value(entry, ('device_id', 'id', 'Device ID'))
        if name or device_id is not None:
            xpus.append(IntelXpuInfo(name or 'Intel XPU', device_id))
    return xpus


def _parse_xpu_smi_discovery_text(output: str) -> List[IntelXpuInfo]:
    xpus = []
    seen = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or 'intel' not in line.lower():
            continue
        parts = [part.strip() for part in line.split('|') if part.strip()]
        for candidate in parts or [line]:
            if 'intel' in candidate.lower() and candidate not in seen:
                seen.add(candidate)
                xpus.append(IntelXpuInfo(candidate))
                break
    return xpus


def _first_entry_value(entry: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_compute_capability(value: str) -> Optional[float]:
    match = re.search(r'(\d+)(?:\.(\d+))?', value or '')
    if match is None:
        return None
    try:
        return float(f'{match.group(1)}.{match.group(2) or "0"}')
    except ValueError:
        return None


def _is_aliyun_pypi_mirror(index_url: Optional[str]) -> bool:
    if not isinstance(index_url, str):
        return False
    normalized = index_url.strip().lower().rstrip('/')
    return normalized in {
        'https://mirrors.aliyun.com/pypi/simple',
        'http://mirrors.aliyun.com/pypi/simple',
    }


def _has_plain_unpinned_torch(requirements: Sequence[str]) -> bool:
    for req_text in requirements:
        req = Requirement(req_text)
        if canonicalize_name(req.name) == 'torch' and not (req.specifier or req.marker or req.url or req.extras):
            return True
    return False


def has_plain_unpinned_torch(requirements: Iterable[str]) -> bool:
    reqs = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
    return _has_plain_unpinned_torch(reqs)


def _rewrite_torch_family_requirements(requirements: Sequence[str], profile: TorchInstallProfile) -> List[str]:
    rewritten = []
    inserted_profile = False
    for req_text in requirements:
        req = Requirement(req_text)
        package_name = canonicalize_name(req.name)
        if package_name in TORCH_FAMILY_PACKAGES:
            if package_name == 'torch' and not inserted_profile:
                rewritten.extend(profile.requirements)
                inserted_profile = True
            continue
        rewritten.append(str(req))
    if not inserted_profile:
        rewritten = [*profile.requirements, *rewritten]
    return list(dict.fromkeys(rewritten))
