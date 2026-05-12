import json
import os
import os.path as osp
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseTranslator, register_translator
from ..base import DEVICE_SELECTOR


MODEL_REPO_ID = "unsloth/gemma-4-E4B-it-GGUF"
MODEL_DIR = "data/models/gemma-4-E4B-it-GGUF"
MODEL_FILES = {
    "Q4_K_M": "gemma-4-E4B-it-Q4_K_M.gguf",
    # The upstream repo publishes the requested Q6_K_M-level quant as Q6_K.
    "Q6_K_M": "gemma-4-E4B-it-Q6_K.gguf",
}
DEFAULT_QUANTIZATION = "Q4_K_M"
QWEN35_MODEL_REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
QWEN35_MODEL_DIR = "data/models/Qwen3.5-9B-GGUF"
QWEN35_MODEL_FILES = {
    "Q4_K_M": "Qwen3.5-9B-Q4_K_M.gguf",
}
RUNTIME_PATH = "data/models/gemma-4-runtime"
WORKER_PATH = Path(__file__).with_name("gemma4_worker.py")
SETUP_COMMAND = "python scripts/setup_gemma4_runtime.py"
GEMMA_KOREAN_STYLE_GUIDE = (
    "For Japanese-to-Korean manga translation, write 자연스러운 한국어 대사. "
    "Prefer fluent Korean dialogue over literal Japanese word order. Preserve character voice, "
    "honorifics, 반말/존댓말 shifts, emotional force, repeated catchphrases, and recurring terms. "
    "Keep speech-bubble wording compact. Render SFX and mimetic words naturally when they carry meaning; "
    "leave iconic sounds short when a Korean equivalent would feel forced. Correct only obvious OCR noise, "
    "especially broken punctuation, duplicated characters, or vertical text that was read in the wrong order."
)
GEMMA_STYLE_GUIDE_PRESETS = {
    "Default": GEMMA_KOREAN_STYLE_GUIDE,
    "Literal but Natural": (
        "For Japanese-to-Korean manga translation, keep the original meaning and sentence beats close to the source, "
        "but write readable Korean. Preserve repeated terms, honorifics, character-specific 말투, and important ambiguity. "
        "Avoid adding interpretation that is not present in the source. Keep speech-bubble wording compact."
    ),
    "Casual Webtoon Korean": (
        "For Japanese-to-Korean manga translation, write lively modern Korean dialogue that feels natural in a webtoon. "
        "Favor concise 반말/존댓말 choices, idiomatic reactions, and emotional readability over literal Japanese word order. "
        "Keep recurring terms consistent and avoid making character voices too similar."
    ),
    "Formal Polished Korean": (
        "For Japanese-to-Korean manga translation, write polished Korean with clean sentence flow and restrained slang. "
        "Preserve speech level, honorifics, characterization, and emotional nuance. Keep dialogue compact enough for speech bubbles "
        "and avoid overly casual phrasing unless the character voice clearly calls for it."
    ),
}


GEMMA_LANG_MAP = {
    "简体中文": "Simplified Chinese",
    "繁體中文": "Traditional Chinese",
    "日本語": "Japanese",
    "English": "English",
    "한국어": "Korean",
    "Tiếng Việt": "Vietnamese",
    "čeština": "Czech",
    "Nederlands": "Dutch",
    "Français": "French",
    "Deutsch": "German",
    "magyar nyelv": "Hungarian",
    "Italiano": "Italian",
    "Polski": "Polish",
    "Português": "Portuguese",
    "Brazilian Portuguese": "Brazilian Portuguese",
    "limba română": "Romanian",
    "русский язык": "Russian",
    "Español": "Spanish",
    "Türk dili": "Turkish",
    "украї́нська мо́ва": "Ukrainian",
    "Thai": "Thai",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Malayalam": "Malayalam",
    "Tamil": "Tamil",
}


class LocalGGUFTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    model_log_name = "Local GGUF"
    model_repo_id = ""
    model_dir = ""
    model_files: Dict[str, str] = {}
    default_quantization = DEFAULT_QUANTIZATION
    runtime_path = RUNTIME_PATH
    worker_path = WORKER_PATH
    worker_python_env_var = "BALLOONTRANS_GEMMA4_PYTHON"
    setup_command = SETUP_COMMAND
    download_command = ""

    def __init__(self, *args, **kwargs) -> None:
        self.params = deepcopy(type(self).params)
        super().__init__(*args, **kwargs)

    def _setup_translator(self):
        self.lang_map.update(GEMMA_LANG_MAP)

    @property
    def thinking_mode(self) -> bool:
        return bool(self.get_param_value("thinking mode"))

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        if self.params is not None and "top_p" in self.params:
            return float(self.get_param_value("top_p"))
        return float(self.get_param_value("top p"))

    @property
    def top_k(self) -> int:
        if self.params is not None and "top_k" in self.params:
            return int(self.get_param_value("top_k"))
        return int(self.get_param_value("top k"))

    @property
    def worker_timeout(self) -> int:
        return int(self.get_param_value("worker timeout"))

    @property
    def model_quantization(self) -> str:
        quantization = self.get_param_value("model quantization")
        if quantization not in type(self).model_files:
            return type(self).default_quantization
        return quantization

    @property
    def model_filename(self) -> str:
        return type(self).model_files[self.model_quantization]

    @property
    def model_path(self) -> str:
        return str(Path(type(self).model_dir) / self.model_filename)

    def _resolve_worker_python(self) -> Optional[str]:
        configured = self.get_param_value("worker python")
        if configured:
            return configured

        env_python = os.environ.get(type(self).worker_python_env_var)
        if env_python:
            return env_python

        runtime_dir = Path(type(self).runtime_path)
        candidates = [
            runtime_dir / "bin" / "python",
            runtime_dir / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _subprocess_error_translations(self, src_list: List[str], message: str) -> List[str]:
        return [
            "" if not isinstance(source_text, str) or not source_text.strip()
            else f"[ERROR: {message}]"
            for source_text in src_list
        ]

    def _gpu_layers(self) -> int:
        if self.get_param_value("device") == "cpu":
            return 0
        return int(self.get_param_value("gpu layers"))

    def _optional_param_value(self, param_key: str, default):
        if self.params is not None and param_key in self.params:
            return self.get_param_value(param_key)
        return default

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        model_path = self.model_path
        model_filename = self.model_filename
        model_quantization = self.model_quantization

        if not osp.isfile(model_path):
            return self._subprocess_error_translations(
                src_list,
                (
                    f"{type(self).model_log_name} {model_quantization} model file is missing: {model_path}. "
                    f"Download {type(self).model_repo_id}/{model_filename} to {type(self).model_dir}."
                    + (f" You can run `{type(self).download_command}`." if type(self).download_command else "")
                ),
            )

        worker_python = self._resolve_worker_python()
        if not worker_python:
            return self._subprocess_error_translations(
                src_list,
                (
                    f"{type(self).model_log_name} runtime is not configured. "
                    f"Run `{type(self).setup_command}` once, or set `worker python`."
                ),
            )

        payload = {
            "model_path": model_path,
            "model_quantization": model_quantization,
            "model_log_name": type(self).model_log_name,
            "texts": src_list,
            "source_lang": self.lang_map[self.lang_source],
            "target_lang": self.lang_map[self.lang_target],
            "max_input_tokens": int(self.get_param_value("max input tokens")),
            "max_new_tokens": int(self.get_param_value("max new tokens")),
            "context_tokens": int(self.get_param_value("context tokens")),
            "gpu_layers": self._gpu_layers(),
            "threads": int(self.get_param_value("threads")),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "thinking_mode": self.thinking_mode,
            "structure_retry_count": int(self._optional_param_value("structure retry count", 1)),
            "chunk_context_cells": int(self._optional_param_value("chunk context cells", 2)),
            "style_guide": str(self._optional_param_value("style guide", "")),
        }

        try:
            proc = subprocess.run(
                [worker_python, str(type(self).worker_path)],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.worker_timeout,
                check=False,
            )
        except Exception as exc:
            self.logger.error(f"{type(self).model_log_name} subprocess failed to start: {exc}")
            return self._subprocess_error_translations(
                src_list,
                f"{type(self).model_log_name} subprocess failed to start: {exc}",
            )

        if proc.returncode != 0:
            self.logger.error(f"{type(self).model_log_name} subprocess failed with code {proc.returncode}: {proc.stderr}")
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
            return self._subprocess_error_translations(src_list, f"{type(self).model_log_name} subprocess failed: {err}")

        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self.logger.error(f"{type(self).model_log_name} subprocess returned invalid JSON: {proc.stdout}\nSTDERR:\n{proc.stderr}")
            return self._subprocess_error_translations(
                src_list,
                f"{type(self).model_log_name} subprocess returned invalid JSON: {exc}",
            )

        translations = response.get("translations")
        if not isinstance(translations, list) or len(translations) != len(src_list):
            self.logger.error(f"{type(self).model_log_name} subprocess returned invalid translation payload: {response}")
            return self._subprocess_error_translations(
                src_list,
                f"{type(self).model_log_name} subprocess returned invalid translation payload.",
            )

        return [text if isinstance(text, str) else "" for text in translations]


@register_translator("Gemma 4 E4B-it")
class Gemma4E4BTranslator(LocalGGUFTranslator):
    model_log_name = "Gemma4 GGUF"
    model_repo_id = MODEL_REPO_ID
    model_dir = MODEL_DIR
    model_files = MODEL_FILES
    default_quantization = DEFAULT_QUANTIZATION
    download_command = "python scripts/setup_gemma4_runtime.py --download-model"
    hf_model_repo_id = MODEL_REPO_ID
    hf_model_save_dir = MODEL_DIR
    hf_model_required_files = [list(MODEL_FILES.values())]
    hf_model_allow_patterns = [MODEL_FILES[DEFAULT_QUANTIZATION], "README.md", "*.json", "*.jinja"]

    params: Dict = {
        "description": (
            "Offline Gemma 4 E4B-it translator using unsloth/gemma-4-E4B-it-GGUF "
            "Q4_K_M or Q6_K_M. Place the selected GGUF file in data/models/gemma-4-E4B-it-GGUF."
        ),
        "device": DEVICE_SELECTOR(),
        "model quantization": {
            "type": "selector",
            "options": list(MODEL_FILES.keys()),
            "value": DEFAULT_QUANTIZATION,
            "description": "GGUF quantization. Q6_K_M uses the upstream gemma-4-E4B-it-Q6_K.gguf file.",
        },
        "worker python": {
            "value": "",
            "description": "Optional Python executable for the isolated Gemma4 GGUF runtime. Empty uses BALLOONTRANS_GEMMA4_PYTHON or data/models/gemma-4-runtime.",
        },
        "worker timeout": {
            "value": 600,
            "description": "Maximum seconds for one Gemma4 GGUF subprocess translation call.",
        },
        "low vram mode": {
            "type": "checkbox",
            "value": True,
            "description": "Gemma4 GGUF runs in a subprocess, so model memory is released after each translation call.",
        },
        "max input tokens": {
            "value": 4096,
            "description": "Target prompt budget for the full page text list.",
        },
        "max new tokens": {
            "value": 2048,
            "description": "Maximum generated tokens for the full page translation response.",
        },
        "context tokens": {
            "value": 8192,
            "description": "llama.cpp context size for the GGUF model.",
        },
        "gpu layers": {
            "value": -1,
            "description": "llama.cpp n_gpu_layers. Use 0 for CPU only, -1 to offload all supported layers.",
        },
        "threads": {
            "value": 0,
            "description": "llama.cpp CPU threads. 0 lets llama.cpp choose automatically.",
        },
        "temperature": {
            "value": 0.15,
            "description": "Sampling temperature. A small value can improve natural dialogue while staying stable.",
        },
        "top_p": {
            "value": 1.0,
            "description": "Nucleus sampling top_p. Lower values narrow token choices; 1.0 disables nucleus filtering.",
        },
        "top_k": {
            "value": 40,
            "description": "Top-k sampling limit. 0 disables top-k filtering in llama.cpp.",
        },
        "thinking mode": {
            "type": "checkbox",
            "value": True,
            "description": "Allow Gemma thinking behavior in the prompt. Output is still constrained to page translations only.",
        },
        "structure retry count": {
            "value": 1,
            "description": "Number of strict JSON/schema retries before splitting a failed Gemma chunk.",
        },
        "chunk context cells": {
            "value": 2,
            "description": "Number of neighboring page cells included as context-only text around split chunks or repair retries.",
        },
        "style guide": {
            "type": "editor",
            "value": GEMMA_KOREAN_STYLE_GUIDE,
            "hidden": True,
            "description": "Optional Gemma translation style guide. Empty uses the worker default.",
        },
        "style guide presets": {
            "type": "style_guide_manager",
            "value": {
                "selected": "Default",
                "styles": GEMMA_STYLE_GUIDE_PRESETS,
            },
            "description": "Select, add, replace, or delete reusable Gemma translation style guides.",
        },
    }


@register_translator("Qwen3.5 9B GGUF")
class Qwen35NineBGGUFTranslator(LocalGGUFTranslator):
    model_log_name = "Qwen3.5 GGUF"
    model_repo_id = QWEN35_MODEL_REPO_ID
    model_dir = QWEN35_MODEL_DIR
    model_files = QWEN35_MODEL_FILES
    default_quantization = DEFAULT_QUANTIZATION
    setup_command = "python scripts/setup_gemma4_runtime.py --model qwen3.5"
    download_command = "python scripts/setup_gemma4_runtime.py --model qwen3.5 --download-model"
    hf_model_repo_id = QWEN35_MODEL_REPO_ID
    hf_model_save_dir = QWEN35_MODEL_DIR
    hf_model_required_files = [list(QWEN35_MODEL_FILES.values())]
    hf_model_allow_patterns = [QWEN35_MODEL_FILES[DEFAULT_QUANTIZATION], "README.md", "*.json", "*.jinja"]

    params: Dict = {
        "description": (
            "Offline Qwen3.5 9B translator using unsloth/Qwen3.5-9B-GGUF "
            "Q4_K_M. Place Qwen3.5-9B-Q4_K_M.gguf in data/models/Qwen3.5-9B-GGUF."
        ),
        "device": DEVICE_SELECTOR(),
        "model quantization": {
            "type": "selector",
            "options": list(QWEN35_MODEL_FILES.keys()),
            "value": DEFAULT_QUANTIZATION,
            "description": "GGUF quantization for unsloth/Qwen3.5-9B-GGUF.",
        },
        "worker python": {
            "value": "",
            "description": "Optional Python executable for the isolated GGUF runtime. Empty uses BALLOONTRANS_GEMMA4_PYTHON or data/models/gemma-4-runtime.",
        },
        "worker timeout": {
            "value": 600,
            "description": "Maximum seconds for one Qwen3.5 GGUF subprocess translation call.",
        },
        "low vram mode": {
            "type": "checkbox",
            "value": True,
            "description": "Qwen3.5 GGUF runs in a subprocess, so model memory is released after each translation call.",
        },
        "max input tokens": {
            "value": 4096,
            "description": "Target prompt budget for the full page text list.",
        },
        "max new tokens": {
            "value": 2048,
            "description": "Maximum generated tokens for the full page translation response.",
        },
        "context tokens": {
            "value": 8192,
            "description": "llama.cpp context size for the GGUF model.",
        },
        "gpu layers": {
            "value": -1,
            "description": "llama.cpp n_gpu_layers. Use 0 for CPU only, -1 to offload all supported layers.",
        },
        "threads": {
            "value": 0,
            "description": "llama.cpp CPU threads. 0 lets llama.cpp choose automatically.",
        },
        "temperature": {
            "value": 0.0,
            "description": "Sampling temperature. 0 keeps translation deterministic.",
        },
        "top_p": {
            "value": 1.0,
            "description": "Nucleus sampling top_p. Lower values narrow token choices; 1.0 disables nucleus filtering.",
        },
        "top_k": {
            "value": 40,
            "description": "Top-k sampling limit. 0 disables top-k filtering in llama.cpp.",
        },
        "thinking mode": {
            "type": "checkbox",
            "value": True,
            "description": "Allow Qwen thinking behavior in the prompt. Output is still constrained to page translations only.",
        },
    }
