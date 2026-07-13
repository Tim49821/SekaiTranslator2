import ast
import importlib.metadata
import os
import platform
import re
import sys
import tempfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.registry import ModuleSpec
from utils.torch_install_helper import detect_nvidia_gpus

from .base import MODULE_ROOT, MODULE_SCRIPTS


UNKNOWN = object()

DECORATORS = {
    'translator': {'register_translator'},
    'textdetector': {'register_textdetectors'},
    'inpainter': {'register_inpainter'},
    'ocr': {'register_OCR'},
}
MODULE_METADATA_ATTRS = {
    'params',
    'lazy_params',
    'download_file_list',
    'download_file_on_load',
    'dependencies',
    'hf_model_repo_id',
    'hf_model_save_dir',
    'hf_model_required_files',
    'hf_model_allow_patterns',
    'hf_model_ignore_patterns',
    'hf_model_download_on_prepare',
    'hf_model_revision',
}
EXTRA_MODULE_FILES = {
    'translator': [str(MODULE_ROOT / 'translators' / 'base.py')],
    'inpainter': [str(MODULE_ROOT / 'inpaint' / 'base.py')],
}
PACKAGE_ROOT = MODULE_ROOT.parent
INITIALIZED_REGISTRIES = set()

BASE_TRANSLATOR_LANGS = [
    'Auto', '简体中文', '繁體中文', '日本語', 'English', '한국어',
    'Tiếng Việt', 'Français', 'Deutsch', 'Italiano', 'Português',
    'русский язык', 'Español', 'Thai', 'Arabic', 'Hindi',
]
DEFAULT_LLM_PROVIDER_MODEL_OPTIONS = {
    'OpenAI': ['OAI: gpt-5.2', 'OAI: gpt-5-mini', 'OAI: gpt-5-nano'],
    'Google': [
        'GGL: gemini-3.1-pro-preview',
        'GGL: gemini-3-flash-preview',
        'GGL: gemini-3.1-flash-lite',
    ],
    'Grok': ['XAI: grok-4', 'XAI: grok-3', 'XAI: grok-3-mini'],
    'OpenRouter': ['LLMS: (override model field)'],
    'LLM Studio': ['LLMS: (override model field)'],
}
DEFAULT_LLM_PROVIDER_DEFAULT_MODELS = {
    'OpenAI': 'OAI: gpt-5.2',
    'Google': 'GGL: gemini-3.1-pro-preview',
    'Grok': 'XAI: grok-4',
    'OpenRouter': 'LLMS: (override model field)',
    'LLM Studio': 'LLMS: (override model field)',
}


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_package_backend():
    if _nvidia_cuda_available():
        return 'cuda'
    version = _package_version('torch')
    if version is None:
        return None
    if sys.platform == 'darwin':
        return 'mps'
    if '+' not in version:
        return None
    local_version = version.split('+', 1)[1].lower()
    if local_version.startswith(('cu', 'rocm')):
        return 'cuda'
    if local_version.startswith('xpu'):
        return 'xpu'
    return None


@lru_cache(maxsize=1)
def _nvidia_cuda_available() -> bool:
    if sys.platform not in {'win32', 'linux'}:
        return False
    return bool(detect_nvidia_gpus())


def _candidate_device_options():
    options = ['cpu']
    backend = _torch_package_backend()
    if backend is not None:
        options.append(backend)
    return options


def _preferred_device_value(options):
    preferred = ['mps'] if sys.platform == 'darwin' else ['cuda', 'xpu']
    for device in preferred:
        if device in options:
            return device
    return 'cpu' if 'cpu' in options else (options[0] if options else 'cpu')


def _device_selector(not_supported=None):
    if not_supported is None:
        not_supported = []
    options = [
        opt for opt in _candidate_device_options()
        if all(device not in opt for device in not_supported)
    ]
    return {
        'type': 'selector',
        'options': options,
        'value': _preferred_device_value(options),
        '__device_not_supported': list(not_supported),
    }


def _find_model_paths(model_dir, prefixes):
    default_path_list = [
        'data/models/ysgyolo_yolo26_2.0.pt',
        'data/models/ysgyolo_yolo26OBB_2.0.pt',
    ]
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    try:
        names = sorted(os.listdir(model_dir))
    except OSError:
        return default_path_list
    found_list = [
        os.path.join(model_dir, name).replace('\\', '/')
        for name in names
        if name.startswith(tuple(prefixes))
    ]
    for path in default_path_list:
        if path not in found_list:
            found_list.append(path)
    return found_list


class SafeEval:
    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def eval(self, node):
        try:
            return self.visit(node)
        except Exception:
            return UNKNOWN

    def visit(self, node):
        visitor = getattr(self, 'visit_' + node.__class__.__name__, None)
        if visitor is None:
            return UNKNOWN
        return visitor(node)

    def visit_Constant(self, node):
        return node.value

    def visit_JoinedStr(self, node):
        values = []
        for part in node.values:
            value = self.visit(part)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(str(value))
        return ''.join(values)

    def visit_FormattedValue(self, node):
        value = self.visit(node.value)
        if value is UNKNOWN:
            return UNKNOWN
        if node.conversion == ord('r'):
            value = repr(value)
        elif node.conversion == ord('a'):
            value = ascii(value)
        else:
            value = str(value)
        if node.format_spec is None:
            return value
        format_spec = self.visit(node.format_spec)
        if format_spec is UNKNOWN:
            return UNKNOWN
        return format(value, format_spec)

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id == 'None':
            return None
        return UNKNOWN

    def visit_List(self, node):
        values = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                value = self.visit(item.value)
                if value is UNKNOWN:
                    return UNKNOWN
                values.extend(list(value))
                continue
            value = self.visit(item)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(value)
        return values

    def visit_Tuple(self, node):
        values = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                value = self.visit(item.value)
                if value is UNKNOWN:
                    return UNKNOWN
                values.extend(list(value))
                continue
            value = self.visit(item)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(value)
        return tuple(values)

    def visit_Set(self, node):
        values = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                value = self.visit(item.value)
                if value is UNKNOWN:
                    return UNKNOWN
                values.extend(list(value))
                continue
            value = self.visit(item)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(value)
        return set(values)

    def visit_Dict(self, node):
        out = {}
        for key_node, value_node in zip(node.keys, node.values):
            value = self.visit(value_node)
            if value is UNKNOWN:
                return UNKNOWN
            if key_node is None:
                if not isinstance(value, dict):
                    return UNKNOWN
                out.update(value)
                continue
            key = self.visit(key_node)
            if key is UNKNOWN:
                return UNKNOWN
            out[key] = value
        return out

    def visit_ListComp(self, node):
        values = []
        saved_env = dict(self.env)

        def assign_target(target, value):
            if isinstance(target, ast.Name):
                self.env[target.id] = value
                return True
            if isinstance(target, (ast.Tuple, ast.List)):
                try:
                    value_list = list(value)
                except TypeError:
                    return False
                if len(target.elts) != len(value_list):
                    return False
                return all(assign_target(subtarget, subvalue) for subtarget, subvalue in zip(target.elts, value_list))
            return False

        def walk_generator(index: int):
            if index >= len(node.generators):
                value = self.visit(node.elt)
                if value is UNKNOWN:
                    return False
                values.append(value)
                return True

            generator = node.generators[index]
            iterable = self.visit(generator.iter)
            if iterable is UNKNOWN:
                return False
            for item in iterable:
                if not assign_target(generator.target, item):
                    return False
                include = True
                for cond in generator.ifs:
                    cond_value = self.visit(cond)
                    if cond_value is UNKNOWN:
                        return False
                    include = include and bool(cond_value)
                if include and not walk_generator(index + 1):
                    return False
            return True

        try:
            if not walk_generator(0):
                return UNKNOWN
            return values
        finally:
            self.env.clear()
            self.env.update(saved_env)

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if value is UNKNOWN:
            return UNKNOWN
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
        return UNKNOWN

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            for value in values:
                if value is False:
                    return False
                if value is UNKNOWN:
                    return UNKNOWN
            return True
        if isinstance(node.op, ast.Or):
            for value in values:
                if value is True:
                    return True
                if value is UNKNOWN:
                    return UNKNOWN
            return False
        return UNKNOWN

    def visit_Compare(self, node):
        left = self.visit(node.left)
        if left is UNKNOWN:
            return UNKNOWN
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if right is UNKNOWN:
                return UNKNOWN
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            elif isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            else:
                return UNKNOWN
            if not ok:
                return False
            left = right
        return True

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mod):
            return left % right
        return UNKNOWN

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        if test is UNKNOWN:
            return self.visit(node.orelse)
        return self.visit(node.body if test else node.orelse)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if value is UNKNOWN or index is UNKNOWN:
            return UNKNOWN
        try:
            return value[index]
        except Exception:
            return UNKNOWN

    def visit_Slice(self, node):
        lower = None if node.lower is None else self.visit(node.lower)
        upper = None if node.upper is None else self.visit(node.upper)
        step = None if node.step is None else self.visit(node.step)
        if lower is UNKNOWN or upper is UNKNOWN or step is UNKNOWN:
            return UNKNOWN
        return slice(lower, upper, step)

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        if value is UNKNOWN:
            if isinstance(node.value, ast.Name):
                root = node.value.id
                if root == 'sys' and node.attr == 'platform':
                    return sys.platform
                if root == 'shared':
                    if node.attr == 'ON_WINDOWS':
                        return sys.platform == 'win32'
                    if node.attr == 'ON_MACOS':
                        return sys.platform == 'darwin'
                    if node.attr == 'ON_LINUX':
                        return sys.platform.startswith('linux')
            return UNKNOWN
        return getattr(value, node.attr, UNKNOWN)

    def visit_Call(self, node):
        func_name = _call_name(node.func)
        args = [self.visit(arg) for arg in node.args]
        if any(arg is UNKNOWN for arg in args):
            return UNKNOWN

        if func_name == 'platform.system':
            return platform.system()
        if func_name == 'platform.mac_ver':
            return platform.mac_ver()
        if func_name == 'platform.machine':
            return platform.machine()
        if isinstance(node.func, ast.Attribute) and not args and not node.keywords:
            value = self.visit(node.func.value)
            if value is UNKNOWN:
                return UNKNOWN
            if isinstance(value, dict):
                if node.func.attr == 'copy':
                    return value.copy()
                if node.func.attr == 'keys':
                    return list(value.keys())
                if node.func.attr == 'values':
                    return list(value.values())
                if node.func.attr == 'items':
                    return list(value.items())
            if isinstance(value, str):
                if node.func.attr == 'lower':
                    return value.lower()
                if node.func.attr == 'upper':
                    return value.upper()
                if node.func.attr == 'strip':
                    return value.strip()
            if isinstance(value, (list, set)) and node.func.attr == 'copy':
                return value.copy()
            return UNKNOWN

        if func_name == 'DEVICE_SELECTOR':
            not_supported = args[0] if args else []
            for kw in node.keywords:
                if kw.arg == 'not_supported':
                    value = self.visit(kw.value)
                    not_supported = [] if value is UNKNOWN else value
            return _device_selector(not_supported)
        if func_name == '_build_fixed_provider_params' and len(args) == 3:
            base_params = self.env.get('LLM_API_Translator_PARAMS') or self.env.get('LLM_OCR_PARAMS')
            if isinstance(base_params, dict):
                params = deepcopy(base_params)
                params.pop('provider', None)
                if isinstance(params.get('model'), dict):
                    params['model']['options'] = args[1]
                    params['model']['value'] = args[2]
                params['description'] = args[0]
                return params
        if func_name in {'deepcopy', 'copy.deepcopy'} and len(args) == 1:
            return deepcopy(args[0])
        if func_name == 'list' and len(args) == 1:
            return list(args[0])
        if func_name == 'tuple' and len(args) == 1:
            return tuple(args[0])
        if func_name == 'set' and len(args) == 1:
            return set(args[0])
        if func_name == 'str' and len(args) == 1:
            return str(args[0])
        if func_name == 'int' and len(args) == 1:
            return int(args[0])
        if func_name == 'float' and len(args) == 1:
            return float(args[0])
        if func_name in {'os.path.join', 'osp.join'}:
            return os.path.join(*args)
        if func_name == 'find_model_paths' and len(args) == 2:
            return _find_model_paths(*args)
        return UNKNOWN


def _call_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f'{parent}.{node.attr}' if parent else node.attr
    return ''


def _module_name_from_path(path: str) -> str:
    path_obj = Path(path).resolve()
    try:
        rel_path = path_obj.relative_to(PACKAGE_ROOT)
        return '.'.join(rel_path.with_suffix('').parts)
    except ValueError:
        module_name = path.replace(os.sep, '.').replace('/', '.')
        return module_name[:-3] if module_name.endswith('.py') else module_name


def _decorator_key(node, module_type: str, env: Dict[str, Any]) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    if _call_name(node.func) not in DECORATORS[module_type]:
        return None
    if not node.args:
        return None
    value = SafeEval(env).eval(node.args[0])
    return value if isinstance(value, str) else None


def _assign_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None, None


def _walk_assignments(stmts: Iterable[ast.stmt], env: Dict[str, Any]):
    evaluator = SafeEval(env)
    for node in stmts:
        name, value_node = _assign_name(node)
        if name is None or value_node is None:
            continue
        value = evaluator.eval(value_node)
        if value is not UNKNOWN:
            env[name] = value


def _collect_class_attrs(class_node: ast.ClassDef, env: Dict[str, Any]) -> Dict[str, Any]:
    attrs = {}
    warnings = []
    class_env = env.copy()

    def walk(stmts):
        evaluator = SafeEval(class_env)
        for node in stmts:
            name, value_node = _assign_name(node)
            if name is not None and value_node is not None:
                value = evaluator.eval(value_node)
                if value is not UNKNOWN:
                    class_env[name] = value
                    if name in MODULE_METADATA_ATTRS:
                        attrs[name] = value
                elif name in MODULE_METADATA_ATTRS:
                    warnings.append(f'{class_node.name}.{name} could not be evaluated lazily')
            elif isinstance(node, ast.If):
                cond = evaluator.eval(node.test)
                if cond is True:
                    walk(node.body)
                elif cond is False:
                    walk(node.orelse)
                else:
                    walk(node.body)
                    walk(node.orelse)

    walk(class_node.body)
    if 'params' not in attrs and isinstance(attrs.get('lazy_params'), dict):
        attrs['params'] = deepcopy(attrs['lazy_params'])
        warnings = [
            warning for warning in warnings
            if warning != f'{class_node.name}.params could not be evaluated lazily'
        ]
    attrs.pop('lazy_params', None)
    if warnings:
        attrs['__metadata_warnings'] = warnings
    return attrs


def _return_list(func_node: ast.FunctionDef, env: Dict[str, Any]):
    evaluator = SafeEval(env)
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            value = evaluator.eval(node.value)
            if isinstance(value, list):
                return value
    return None


def _is_self_lang_map(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'lang_map'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    )


def _append_lang_if_supported(langs: List[str], key, value) -> bool:
    if isinstance(key, str) and value not in {'', None, UNKNOWN}:
        if key not in langs:
            langs.append(key)
        return True
    return False


def _collect_translator_langs(class_node: ast.ClassDef, env: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[List[str]], List[str]]:
    langs = []
    src = tgt = None
    warnings = []
    cht_require_convert = False
    evaluator = SafeEval(env)

    for node in class_node.body:
        name, value_node = _assign_name(node)
        if name == 'cht_require_convert' and value_node is not None:
            value = evaluator.eval(value_node)
            if isinstance(value, bool):
                cht_require_convert = value

        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in {'supported_src_list', 'supported_tgt_list'}:
            value = _return_list(node, env)
            if node.name == 'supported_src_list':
                src = value
            else:
                tgt = value
        if node.name != '_setup_translator':
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and len(child.targets) == 1:
                target = child.targets[0]
                if _is_self_lang_map(target):
                    mapping = evaluator.eval(child.value)
                    if not isinstance(mapping, dict):
                        warnings.append(f'{class_node.name} has unsupported lazy lang_map assignment')
                        continue
                    for key, value in mapping.items():
                        _append_lang_if_supported(langs, key, value)
                elif (
                    isinstance(target, ast.Subscript)
                    and _is_self_lang_map(target.value)
                ):
                    key = evaluator.eval(target.slice)
                    value = evaluator.eval(child.value)
                    if not _append_lang_if_supported(langs, key, value):
                        warnings.append(f'{class_node.name} has unsupported lazy lang_map assignment')
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                call = child.value
                if not isinstance(call.func, ast.Attribute) or call.func.attr != 'update':
                    continue
                if not _is_self_lang_map(call.func.value):
                    continue
                if len(call.args) != 1:
                    warnings.append(f'{class_node.name} has unsupported lazy lang_map.update call')
                    continue
                mapping = evaluator.eval(call.args[0])
                if isinstance(mapping, dict):
                    for key, value in mapping.items():
                        _append_lang_if_supported(langs, key, value)

    if class_node.name in {'TransNone', 'TransSource'}:
        langs = BASE_TRANSLATOR_LANGS.copy()
    if cht_require_convert and '简体中文' in langs and '繁體中文' not in langs:
        langs.append('繁體中文')
    if src is None:
        src = langs or None
    if tgt is None:
        tgt = langs or None
    return src, tgt, warnings


def validate_lazy_module_specs(specs: Iterable[ModuleSpec]) -> List[str]:
    warnings = []
    for spec in specs:
        for warning in spec.metadata_warnings:
            warnings.append(f'{spec.key}: {warning}')
        if spec.module_type == 'translator' and (not spec.supported_src_list or not spec.supported_tgt_list):
            warnings.append(f'{spec.key}: translator has no lazy supported language metadata')
        for param_key, param in (spec.params or {}).items():
            if not isinstance(param, dict) or param.get('type') != 'selector':
                continue
            options = param.get('options')
            if options is UNKNOWN:
                warnings.append(f'{spec.key}.{param_key}: selector options could not be evaluated lazily')
            elif not options and not (param.get('editable', False) and param.get('value') not in {'', None}):
                warnings.append(f'{spec.key}.{param_key}: selector has no lazy options or editable fallback')
    return warnings


def _scan_file(path: str, module_type: str) -> List[ModuleSpec]:
    with open(path, 'r', encoding='utf8') as f:
        tree = ast.parse(f.read(), filename=path)
    module_path = _module_name_from_path(path)
    specs = []
    env = {
        'sys': sys,
        'platform': platform,
        'DEFAULT_DEVICE': 'cpu',
        'BF16_SUPPORTED': False,
        'LLM_PROVIDER_MODEL_OPTIONS': deepcopy(DEFAULT_LLM_PROVIDER_MODEL_OPTIONS),
        'LLM_PROVIDER_DEFAULT_MODELS': deepcopy(DEFAULT_LLM_PROVIDER_DEFAULT_MODELS),
        'LLM_OCR_PROVIDER_MODEL_OPTIONS': {
            **deepcopy(DEFAULT_LLM_PROVIDER_MODEL_OPTIONS),
            'Ollama': ['OLLAMA: (override model field)'],
        },
        'LLM_OCR_PROVIDER_DEFAULT_MODELS': {
            **deepcopy(DEFAULT_LLM_PROVIDER_DEFAULT_MODELS),
            'Ollama': 'OLLAMA: (override model field)',
        },
        'True': True,
        'False': False,
        'None': None,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'set': set,
        'dict': dict,
    }
    class_attrs_by_name: Dict[str, Dict[str, Any]] = {}
    translator_langs_by_class: Dict[str, Tuple[Optional[List[str]], Optional[List[str]]]] = {}

    def walk(stmts):
        _walk_assignments(stmts, env)
        evaluator = SafeEval(env)
        for node in stmts:
            if isinstance(node, ast.ClassDef):
                collected_attrs = _collect_class_attrs(node, env)
                inherited_attrs = {}
                for base in node.bases:
                    base_name = _call_name(base)
                    if base_name in class_attrs_by_name:
                        inherited_attrs.update(deepcopy(class_attrs_by_name[base_name]))
                attrs_for_class = deepcopy(inherited_attrs)
                attrs_for_class.update(collected_attrs)
                class_attrs_by_name[node.name] = deepcopy(attrs_for_class)
                env[node.name] = SimpleNamespace(**attrs_for_class)
                if 'params' in attrs_for_class:
                    env[f'{node.name}_PARAMS'] = deepcopy(attrs_for_class['params'])

                src = tgt = None
                lang_warnings = []
                if module_type == 'translator':
                    src, tgt, lang_warnings = _collect_translator_langs(node, env)
                    inherited_src = inherited_tgt = None
                    for base in node.bases:
                        base_name = _call_name(base)
                        base_langs = translator_langs_by_class.get(base_name)
                        if base_langs is None:
                            continue
                        base_src, base_tgt = base_langs
                        inherited_src = inherited_src or deepcopy(base_src)
                        inherited_tgt = inherited_tgt or deepcopy(base_tgt)
                    src = src or inherited_src
                    tgt = tgt or inherited_tgt
                    translator_langs_by_class[node.name] = (deepcopy(src), deepcopy(tgt))

                key = None
                for decorator in node.decorator_list:
                    key = _decorator_key(decorator, module_type, env)
                    if key is not None:
                        break
                if key is None:
                    continue
                attrs = attrs_for_class
                metadata_warnings = attrs.get('__metadata_warnings', []).copy()
                if module_type == 'translator':
                    metadata_warnings.extend(lang_warnings)
                specs.append(ModuleSpec(
                    key=key,
                    import_path=module_path,
                    class_name=node.name,
                    module_type=module_type,
                    params=attrs.get('params'),
                    download_file_list=attrs.get('download_file_list'),
                    download_file_on_load=attrs.get('download_file_on_load', False),
                    dependencies=deepcopy(attrs.get('dependencies', [])),
                    hf_model_repo_id=attrs.get('hf_model_repo_id'),
                    hf_model_save_dir=attrs.get('hf_model_save_dir'),
                    hf_model_required_files=deepcopy(attrs.get('hf_model_required_files')),
                    hf_model_allow_patterns=deepcopy(attrs.get('hf_model_allow_patterns')),
                    hf_model_ignore_patterns=deepcopy(attrs.get('hf_model_ignore_patterns')),
                    hf_model_download_on_prepare=attrs.get('hf_model_download_on_prepare', False),
                    hf_model_revision=attrs.get('hf_model_revision'),
                    supported_src_list=src,
                    supported_tgt_list=tgt,
                    metadata_warnings=metadata_warnings,
                ))
            elif isinstance(node, ast.If):
                cond = evaluator.eval(node.test)
                if cond is True:
                    walk(node.body)
                elif cond is False:
                    walk(node.orelse)
                else:
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)

    walk(tree.body)
    return specs


def init_lazy_module_registries(target_modules=None):
    from . import MODULETYPE_TO_REGISTRIES

    def _module_files(module_type: str) -> List[str]:
        script = MODULE_SCRIPTS[module_type]
        module_dir = script['module_dir']
        pattern = re.compile(script['module_pattern'])
        files = []
        if os.path.isdir(module_dir):
            for name in sorted(os.listdir(module_dir)):
                if pattern.match(name):
                    files.append(os.path.join(module_dir, name))
        files.extend(EXTRA_MODULE_FILES.get(module_type, []))
        return [path for path in files if os.path.exists(path)]

    if target_modules is None:
        targets = list(MODULE_SCRIPTS.keys())
    elif isinstance(target_modules, str):
        targets = [target_modules]
    else:
        targets = list(target_modules)

    for module_type in targets:
        if module_type in INITIALIZED_REGISTRIES:
            continue
        registry = MODULETYPE_TO_REGISTRIES[module_type]
        for path in _module_files(module_type):
            for spec in _scan_file(path, module_type):
                registry.register_lazy_module(spec)
        INITIALIZED_REGISTRIES.add(module_type)
