from pathlib import Path
import ast

from . import shared


def _read_pyproject_version(pyproject_path: Path) -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        with pyproject_path.open('rb') as f:
            data = tomllib.load(f)
        version = data.get('project', {}).get('version')
        if isinstance(version, str) and version:
            return version

    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding='utf8').splitlines():
        line = raw_line.strip()
        if line.startswith('[') and line.endswith(']'):
            in_project_section = line == '[project]'
            continue
        if in_project_section and line.startswith('version'):
            key, sep, value = line.partition('=')
            if sep and key.strip() == 'version':
                return value.strip().strip('"\'')
    raise RuntimeError(f'Failed to read project version from {pyproject_path}')


def get_current_version(program_path: str = None) -> str:
    root = Path(program_path or shared.PROGRAM_PATH)
    pyproject_path = root / 'pyproject.toml'
    if pyproject_path.exists():
        return _read_pyproject_version(pyproject_path)
    launch_path = root / 'launch.py'
    if launch_path.exists():
        try:
            launch_source = launch_path.read_text(encoding='utf8')
            tree = ast.parse(launch_source, filename=str(launch_path))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == 'VERSION' for target in node.targets):
                    continue
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
        except Exception:
            pass
        try:
            for raw_line in launch_path.read_text(encoding='utf8').splitlines():
                key, sep, value = raw_line.partition('=')
                if sep and key.strip() == 'VERSION':
                    parsed = ast.literal_eval(value.strip())
                    if isinstance(parsed, str) and parsed:
                        return parsed
        except Exception:
            pass
    try:
        from importlib.metadata import version
        return version('ballontranslator')
    except Exception:
        return '0.0.0'


APP_VERSION = get_current_version()
