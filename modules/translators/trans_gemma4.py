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
RUNTIME_PATH = "data/models/gemma-4-runtime"
WORKER_PATH = Path(__file__).with_name("gemma4_worker.py")
SETUP_COMMAND = "python scripts/setup_gemma4_runtime.py"


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


@register_translator("Gemma 4 E4B-it")
class Gemma4E4BTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
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
            "value": 0.0,
            "description": "Sampling temperature. 0 keeps translation deterministic.",
        },
        "thinking mode": {
            "type": "checkbox",
            "value": False,
            "description": "Allow Gemma thinking behavior in the prompt. Output is still constrained to page translations only.",
        },
    }

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
    def worker_timeout(self) -> int:
        return int(self.get_param_value("worker timeout"))

    @property
    def model_quantization(self) -> str:
        quantization = self.get_param_value("model quantization")
        if quantization not in MODEL_FILES:
            return DEFAULT_QUANTIZATION
        return quantization

    @property
    def model_filename(self) -> str:
        return MODEL_FILES[self.model_quantization]

    @property
    def model_path(self) -> str:
        return str(Path(MODEL_DIR) / self.model_filename)

    def _resolve_worker_python(self) -> Optional[str]:
        configured = self.get_param_value("worker python")
        if configured:
            return configured

        env_python = os.environ.get("BALLOONTRANS_GEMMA4_PYTHON")
        if env_python:
            return env_python

        runtime_dir = Path(RUNTIME_PATH)
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
                    f"Gemma4 GGUF {model_quantization} model file is missing: {model_path}. "
                    f"Download {MODEL_REPO_ID}/{model_filename} to {MODEL_DIR}."
                ),
            )

        worker_python = self._resolve_worker_python()
        if not worker_python:
            return self._subprocess_error_translations(
                src_list,
                f"Gemma4 GGUF runtime is not configured. Run `{SETUP_COMMAND}` once, or set `worker python`.",
            )

        payload = {
            "model_path": model_path,
            "model_quantization": model_quantization,
            "texts": src_list,
            "source_lang": self.lang_map[self.lang_source],
            "target_lang": self.lang_map[self.lang_target],
            "max_input_tokens": int(self.get_param_value("max input tokens")),
            "max_new_tokens": int(self.get_param_value("max new tokens")),
            "context_tokens": int(self.get_param_value("context tokens")),
            "gpu_layers": self._gpu_layers(),
            "threads": int(self.get_param_value("threads")),
            "temperature": self.temperature,
            "thinking_mode": self.thinking_mode,
        }

        try:
            proc = subprocess.run(
                [worker_python, str(WORKER_PATH)],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.worker_timeout,
                check=False,
            )
        except Exception as exc:
            self.logger.error(f"Gemma4 GGUF subprocess failed to start: {exc}")
            return self._subprocess_error_translations(src_list, f"Gemma4 GGUF subprocess failed to start: {exc}")

        if proc.returncode != 0:
            self.logger.error(f"Gemma4 GGUF subprocess failed with code {proc.returncode}: {proc.stderr}")
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
            return self._subprocess_error_translations(src_list, f"Gemma4 GGUF subprocess failed: {err}")

        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self.logger.error(f"Gemma4 GGUF subprocess returned invalid JSON: {proc.stdout}\nSTDERR:\n{proc.stderr}")
            return self._subprocess_error_translations(src_list, f"Gemma4 GGUF subprocess returned invalid JSON: {exc}")

        translations = response.get("translations")
        if not isinstance(translations, list) or len(translations) != len(src_list):
            self.logger.error(f"Gemma4 GGUF subprocess returned invalid translation payload: {response}")
            return self._subprocess_error_translations(src_list, "Gemma4 GGUF subprocess returned invalid translation payload.")

        return [text if isinstance(text, str) else "" for text in translations]
