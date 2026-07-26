import os
import os.path as osp
import re
from typing import Dict, List, Mapping, Optional, Tuple

from . import shared


DOTENV_PATH = osp.join(shared.PROGRAM_PATH, ".env")

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOADED_DOTENV_PATHS = set()

LLM_LOCAL_PROVIDERS = {"LLM Studio", "Ollama"}
LLM_API_KEY_TIERS = ("Free", "Paid")
LLM_PROVIDER_SLUGS = {
    "OpenAI": "OPENAI",
    "Google": "GOOGLE",
    "Grok": "GROK",
    "OpenRouter": "OPENROUTER",
}
LLM_STANDARD_SINGLE_ENVS = {
    "OpenAI": ("OPENAI_API_KEY",),
    "Google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "Grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "OpenRouter": ("OPENROUTER_API_KEY",),
}
LLM_FIXED_TRANSLATOR_PROVIDERS = {
    "LLM OpenAI": "OpenAI",
    "LLM Google": "Google",
    "LLM Grok": "Grok",
    "LLM OpenRouter": "OpenRouter",
    "LLM Studio": "LLM Studio",
}
LLM_FIXED_OCR_PROVIDERS = {
    "LLM OCR OpenAI": "OpenAI",
    "LLM OCR Google": "Google",
    "LLM OCR Grok": "Grok",
    "LLM OCR OpenRouter": "OpenRouter",
    "LLM OCR Studio": "LLM Studio",
    "LLM OCR Ollama": "Ollama",
}
_PLACEHOLDER_SECRET_VALUES = {
    ".",
    "-",
    "your-api-key",
    "your_api_key",
    "paste-api-key-here",
    "paste_api_key_here",
    "paste-key-here",
    "paste_key_here",
}


def _strip_inline_comment(value: str) -> str:
    match = re.search(r"\s+#", value)
    if match:
        return value[: match.start()].rstrip()
    return value


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ""

    quote = value[0]
    if quote in {'"', "'"} and len(value) >= 2 and value[-1] == quote:
        value = value[1:-1]
        if quote == '"':
            value = (
                value.replace(r"\n", "\n")
                .replace(r"\r", "\r")
                .replace(r"\t", "\t")
                .replace(r"\"", '"')
                .replace(r"\\", "\\")
            )
        return value

    return _strip_inline_comment(value)


def _parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not _KEY_RE.match(key):
        return None
    return key, _unquote_env_value(value)


def parse_dotenv(path: str = DOTENV_PATH) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not osp.exists(path):
        return values

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parsed = _parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            values[key] = value
    return values


def load_dotenv(path: str = DOTENV_PATH, override: bool = False) -> bool:
    values = parse_dotenv(path)
    if not values:
        return False

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED_DOTENV_PATHS.add(osp.abspath(path))
    return True


def _quote_env_value(value: str) -> str:
    if re.match(r"^[A-Za-z0-9_@%+=:,./;~^-]*$", value):
        return value
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', r"\"")
        .replace("\n", r"\n")
        .replace("\r", r"\r")
        .replace("\t", r"\t")
        + '"'
    )


def _line_env_key(line: str) -> Optional[str]:
    parsed = _parse_dotenv_line(line)
    if parsed is None:
        return None
    return parsed[0]


def update_dotenv(values: Mapping[str, str], path: str = DOTENV_PATH) -> bool:
    values = {
        key: str(value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }
    if not values:
        return False

    existing_lines = []
    if osp.exists(path):
        with open(path, "r", encoding="utf8") as f:
            existing_lines = f.readlines()

    updated_lines = []
    written_keys = set()
    for line in existing_lines:
        key = _line_env_key(line)
        if key in values:
            updated_lines.append(f"{key}={_quote_env_value(values[key])}\n")
            written_keys.add(key)
        else:
            updated_lines.append(line)

    if updated_lines and updated_lines[-1] and not updated_lines[-1].endswith("\n"):
        updated_lines[-1] += "\n"

    missing_keys = [key for key in values if key not in written_keys]
    if missing_keys and updated_lines and updated_lines[-1].strip():
        updated_lines.append("\n")
    for key in missing_keys:
        updated_lines.append(f"{key}={_quote_env_value(values[key])}\n")

    dotenv_dir = osp.dirname(path)
    if dotenv_dir and not osp.exists(dotenv_dir):
        os.makedirs(dotenv_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf8") as f:
        f.writelines(updated_lines)
    os.replace(tmp_path, path)

    for key, value in values.items():
        os.environ[key] = value
    return True


def _get_param_value(params: Mapping, key: str) -> str:
    if not isinstance(params, Mapping) or key not in params:
        return ""
    value = params[key]
    if isinstance(value, Mapping):
        value = value.get("value", "")
    return value.strip() if isinstance(value, str) else ""


def _usable_secret_value(value: str) -> str:
    value = value.strip() if isinstance(value, str) else ""
    if not value:
        return ""
    if value.lower() in _PLACEHOLDER_SECRET_VALUES:
        return ""
    return value


def _set_param_value(params: Dict, key: str, value: str):
    if key not in params:
        return
    current = params[key]
    if isinstance(current, dict):
        current["value"] = value
    else:
        params[key] = value


def _module_section(module_cfg, section: str):
    if isinstance(module_cfg, Mapping):
        return module_cfg.get(section, {})
    return getattr(module_cfg, section, {})


def _provider_slug(provider: str) -> Optional[str]:
    return LLM_PROVIDER_SLUGS.get(provider)


def _primary_single_env(provider: str, for_ocr: bool = False) -> Optional[str]:
    slug = _provider_slug(provider)
    if not slug:
        return None
    if for_ocr:
        return f"BALLOONTRANS_LLM_OCR_{slug}_API_KEY"
    return f"BALLOONTRANS_LLM_{slug}_API_KEY"


def _primary_multiple_env(provider: str, for_ocr: bool = False) -> Optional[str]:
    slug = _provider_slug(provider)
    if not slug:
        return None
    if for_ocr:
        return f"BALLOONTRANS_LLM_OCR_{slug}_API_KEYS"
    return f"BALLOONTRANS_LLM_{slug}_API_KEYS"


def normalize_llm_api_key_tier(tier: str) -> str:
    if isinstance(tier, str) and tier.strip().lower() == "paid":
        return "Paid"
    return "Free"


def parse_llm_api_keys(value: str) -> List[str]:
    keys: List[str] = []
    seen = set()
    if not isinstance(value, str):
        return keys
    for key in value.replace("\n", ";").split(";"):
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _primary_tier_env(
    provider: str,
    tier: str,
    for_ocr: bool = False,
) -> Optional[str]:
    slug = _provider_slug(provider)
    if not slug:
        return None
    tier_slug = normalize_llm_api_key_tier(tier).upper()
    if for_ocr:
        return f"BALLOONTRANS_LLM_OCR_{slug}_{tier_slug}_API_KEYS"
    return f"BALLOONTRANS_LLM_{slug}_{tier_slug}_API_KEYS"


def _single_env_candidates(provider: str, for_ocr: bool = False) -> Tuple[str, ...]:
    primary = _primary_single_env(provider, for_ocr=for_ocr)
    if not primary:
        return ()

    candidates = [primary]
    if for_ocr:
        generic = _primary_single_env(provider, for_ocr=False)
        if generic:
            candidates.append(generic)
    candidates.extend(LLM_STANDARD_SINGLE_ENVS.get(provider, ()))
    return tuple(candidates)


def _multiple_env_candidates(provider: str, for_ocr: bool = False) -> Tuple[str, ...]:
    primary = _primary_multiple_env(provider, for_ocr=for_ocr)
    if not primary:
        return ()

    candidates = [primary]
    if for_ocr:
        generic = _primary_multiple_env(provider, for_ocr=False)
        if generic:
            candidates.append(generic)
    return tuple(candidates)


def _first_env_value(candidates: Tuple[str, ...]) -> str:
    load_dotenv()
    for env_name in candidates:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def get_llm_single_api_key(provider: str, for_ocr: bool = False) -> str:
    if provider in LLM_LOCAL_PROVIDERS:
        return ""
    return _first_env_value(_single_env_candidates(provider, for_ocr=for_ocr))


def get_llm_multiple_api_keys(provider: str, for_ocr: bool = False) -> str:
    if provider in LLM_LOCAL_PROVIDERS:
        return ""
    return _first_env_value(_multiple_env_candidates(provider, for_ocr=for_ocr))


def get_llm_api_key_pool(
    provider: str,
    tier: str = "Free",
    for_ocr: bool = False,
) -> List[str]:
    if provider in LLM_LOCAL_PROVIDERS:
        return []

    normalized_tier = normalize_llm_api_key_tier(tier)
    tier_env = _primary_tier_env(
        provider,
        normalized_tier,
        for_ocr=for_ocr,
    )
    tier_value = _first_env_value((tier_env,)) if tier_env else ""
    if tier_value:
        return parse_llm_api_keys(tier_value)
    if normalized_tier == "Paid":
        return []

    single = get_llm_single_api_key(provider, for_ocr=for_ocr)
    multiple = get_llm_multiple_api_keys(provider, for_ocr=for_ocr)
    return parse_llm_api_keys(";".join(value for value in (single, multiple) if value))


def _translator_provider(module_name: str, params: Mapping) -> Optional[str]:
    if module_name in LLM_FIXED_TRANSLATOR_PROVIDERS:
        return LLM_FIXED_TRANSLATOR_PROVIDERS[module_name]
    if module_name == "LLM_API_Translator":
        return _get_param_value(params, "provider") or "OpenAI"
    return None


def _ocr_provider(module_name: str, params: Mapping) -> Optional[str]:
    if module_name in LLM_FIXED_OCR_PROVIDERS:
        return LLM_FIXED_OCR_PROVIDERS[module_name]
    if module_name == "llm_ocr":
        return _get_param_value(params, "provider") or "OpenAI"
    return None


def _collect_llm_api_keys(module_cfg) -> Dict[str, str]:
    env_values: Dict[str, str] = {}

    for module_name, params in _module_section(module_cfg, "translator_params").items():
        provider = _translator_provider(module_name, params)
        if not provider or provider in LLM_LOCAL_PROVIDERS:
            continue

        single = _usable_secret_value(_get_param_value(params, "apikey"))
        multiple = _usable_secret_value(_get_param_value(params, "multiple_keys"))
        free = _usable_secret_value(_get_param_value(params, "free_api_keys"))
        paid = _usable_secret_value(_get_param_value(params, "paid_api_keys"))
        single_env = _primary_single_env(provider)
        multiple_env = _primary_multiple_env(provider)
        free_env = _primary_tier_env(provider, "Free")
        paid_env = _primary_tier_env(provider, "Paid")
        if single and single_env:
            env_values[single_env] = single
        if multiple and multiple_env:
            env_values[multiple_env] = multiple
        free_pool = parse_llm_api_keys(free or ";".join((single, multiple)))
        paid_pool = parse_llm_api_keys(paid)
        if free_pool and free_env:
            env_values[free_env] = ";".join(free_pool)
        if paid_pool and paid_env:
            env_values[paid_env] = ";".join(paid_pool)

    for module_name, params in _module_section(module_cfg, "ocr_params").items():
        provider = _ocr_provider(module_name, params)
        if not provider or provider in LLM_LOCAL_PROVIDERS:
            continue

        single = _usable_secret_value(_get_param_value(params, "api_key"))
        multiple = _usable_secret_value(_get_param_value(params, "multiple_keys"))
        free = _usable_secret_value(_get_param_value(params, "free_api_keys"))
        paid = _usable_secret_value(_get_param_value(params, "paid_api_keys"))
        single_env = _primary_single_env(provider, for_ocr=True)
        multiple_env = _primary_multiple_env(provider, for_ocr=True)
        free_env = _primary_tier_env(provider, "Free", for_ocr=True)
        paid_env = _primary_tier_env(provider, "Paid", for_ocr=True)
        if single and single_env:
            env_values[single_env] = single
        if multiple and multiple_env:
            env_values[multiple_env] = multiple
        free_pool = parse_llm_api_keys(free or ";".join((single, multiple)))
        paid_pool = parse_llm_api_keys(paid)
        if free_pool and free_env:
            env_values[free_env] = ";".join(free_pool)
        if paid_pool and paid_env:
            env_values[paid_env] = ";".join(paid_pool)

    return env_values


def persist_llm_api_keys_from_config(module_cfg, dotenv_path: str = DOTENV_PATH) -> bool:
    return update_dotenv(_collect_llm_api_keys(module_cfg), path=dotenv_path)


def sanitize_llm_api_keys(module_cfg: Dict) -> Dict:
    translator_params = module_cfg.get("translator_params", {})
    for module_name, params in translator_params.items():
        if _translator_provider(module_name, params):
            _set_param_value(params, "apikey", "")
            _set_param_value(params, "multiple_keys", "")
            _set_param_value(params, "free_api_keys", "")
            _set_param_value(params, "paid_api_keys", "")

    for module_name, params in module_cfg.get("ocr_params", {}).items():
        if isinstance(params, dict) and _ocr_provider(module_name, params):
            _set_param_value(params, "api_key", "")
            _set_param_value(params, "multiple_keys", "")
            _set_param_value(params, "free_api_keys", "")
            _set_param_value(params, "paid_api_keys", "")
    return module_cfg
