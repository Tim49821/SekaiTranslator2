import re
import time
import base64
import json
import cv2
import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional

import openai
import httpx

from .base import register_OCR, OCRBase, TextBlock
from utils.env import (
    LLM_API_KEY_TIERS,
    get_llm_api_key_pool,
    normalize_llm_api_key_tier,
    parse_llm_api_keys,
)
from ..translators.trans_llm_api_json import (
    GEMINI_REASONING_EFFORT_OPTIONS,
    LLM_PROVIDER_DEFAULT_MODELS,
    LLM_PROVIDER_MODEL_OPTIONS,
    apply_google_reasoning_effort,
)

LLM_OCR_PROVIDER_MODEL_OPTIONS = {
    provider: list(models)
    for provider, models in LLM_PROVIDER_MODEL_OPTIONS.items()
}
LLM_OCR_PROVIDER_MODEL_OPTIONS.update({
    "Ollama": ["OLLAMA: (override model field)"],
})

LLM_OCR_PROVIDER_DEFAULT_MODELS = {
    provider: model
    for provider, model in LLM_PROVIDER_DEFAULT_MODELS.items()
}
LLM_OCR_PROVIDER_DEFAULT_MODELS.update({
    "Ollama": "OLLAMA: (override model field)",
})

LLM_OCR_PROVIDER_DESCRIPTIONS = {
    "OpenAI": "OpenAI-backed vision LLM OCR.",
    "Google": "Google Gemini-compatible vision LLM OCR.",
    "Grok": "xAI Grok-backed vision LLM OCR.",
    "OpenRouter": "OpenRouter-backed vision LLM OCR.",
    "LLM Studio": "Local LLM Studio-compatible vision LLM OCR.",
    "Ollama": "Local Ollama-compatible vision LLM OCR.",
}

class LLM_OCR(OCRBase):
    lang_map = {
        "Auto Detect": None,
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Basque": "eu",
        "Belarusian": "be",
        "Bengali": "bn",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Cherokee": "chr",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Corsican": "co",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Faroese": "fo",
        "Filipino": "fil",
        "Finnish": "fi",
        "French": "fr",
        "Frisian": "fy",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Indonesian": "id",
        "Interlingua": "ia",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Khmer": "km",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Occitan": "oc",
        "Oriya": "or",
        "Pashto": "ps",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russian": "ru",
        "Samoan": "sm",
        "Scots Gaelic": "gd",
        "Serbian (Cyrillic)": "sr-Cyrl",
        "Serbian (Latin)": "sr-Latn",
        "Shona": "sn",
        "Sindhi": "sd",
        "Sinhala": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tagalog": "tl",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Thai": "th",
        "Tibetan": "bo",
        "Tigrinya": "ti",
        "Tongan": "to",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu",
    }

    popular_models = [
        model
        for provider in ("OpenAI", "Google")
        for model in LLM_OCR_PROVIDER_MODEL_OPTIONS[provider]
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": list(LLM_OCR_PROVIDER_MODEL_OPTIONS.keys()),
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "api_key_tier": {
            "type": "selector",
            "options": list(LLM_API_KEY_TIERS),
            "value": "Free",
            "display_name": "API key tier",
            "description": "Choose which API key pool this OCR module uses.",
        },
        "free_api_keys": {
            "type": "editor",
            "value": "",
            "display_name": "Free API keys",
            "description": (
                "Free API keys separated by semicolons or newlines. "
                "Stored .env keys are loaded here; clearing the field "
                "explicitly disables this pool."
            ),
        },
        "paid_api_keys": {
            "type": "editor",
            "value": "",
            "display_name": "Paid API keys",
            "description": (
                "Paid API keys separated by semicolons or newlines. "
                "Stored .env keys are loaded here; clearing the field "
                "explicitly disables this pool."
            ),
        },
        "__api_key_pool_dirty": {
            "value": "",
            "hidden": True,
        },
        "api_key": {
            "value": "",
            "hidden": True,
            "description": "Legacy single API key, treated as Free.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "hidden": True,
            "description": "Legacy API keys, treated as Free.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "model": {
            "type": "selector",
            "options": [
                model
                for models in LLM_OCR_PROVIDER_MODEL_OPTIONS.values()
                for model in models
            ],
            "value": LLM_OCR_PROVIDER_DEFAULT_MODELS["OpenAI"],
            "description": "Select the model to use.",
        },
        "override_model": {
            "value": "",
            "description": "Specify a custom model name to override the selected one.",
        },
        "language": {
            "type": "selector",
            "options": list(lang_map.keys()),
            "value": "Japanese",
            "description": "Language for OCR.",
        },
        "detail_level": {
            "type": "selector",
            "options": ["auto", "low", "high"],
            "value": "auto",
            "description": "Controls image detail level for vision models.",
        },
        "prompt": {
            "type": "editor",
            "value": "Perform OCR on the provided manga image snippet. The language is **{language}**.\nRecognize all text, including handwritten sound effects (SFX).\n**CRITICAL INSTRUCTION:** If you see jumbled characters, it is likely vertical text that was read horizontally. First, mentally reconstruct the correct vertical text.\n**OUTPUT FORMATTING:** All recognized text from the image must be consolidated into a **single, continuous horizontal line**. Do not use newlines.\nYour final output must be ONLY the recognized text. No explanations.",
            "description": "The main prompt for the OCR task. Use {language} placeholder.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are a specialized OCR engine for manga and comics. Your primary function is to accurately extract and consolidate all recognized text from an image into a **single, continuous horizontal line**. You must return only the raw, recognized text. You do not interpret, translate, or explain the content. You are designed to intelligently handle common OCR errors, such as reconstructing jumbled characters that result from misreading vertical text.",
            "description": "Optional system prompt to guide the model's behavior.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port)",
        },
        "delay": {"value": 1.0, "description": "Delay in seconds between requests."},
        "requests_per_minute": {
            "value": 15,
            "description": "Maximum number of requests per minute per key.",
        },
        "max_response_tokens": {
            "value": 4096,
            "description": "Maximum number of tokens in the LLM's response.",
        },
        "description": "OCR using various vision-capable LLMs.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.current_key_index = 0

    def _initialize_client(self, api_key_to_use: str):
        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Grok":
                endpoint = "https://api.x.ai/v1"
            elif provider == "Ollama":
                endpoint = "http://localhost:11434/v1"

        http_client = None
        if self.proxy:
            try:
                proxy_mounts = {"all://": httpx.HTTPTransport(proxy=self.proxy)}
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{self.proxy}': {e}.")

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        self.client = openai.OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=http_client
        )

    # --- Property Getters (similar to translator) ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def api_key_tier(self) -> str:
        return normalize_llm_api_key_tier(
            self.get_param_value("api_key_tier")
        )

    @property
    def active_api_keys(self) -> List[str]:
        param_key = f"{self.api_key_tier.lower()}_api_keys"
        configured_keys = parse_llm_api_keys(
            self.get_param_value(param_key)
        )
        if configured_keys:
            return configured_keys
        dirty_pools = parse_llm_api_keys(
            self.get_param_value("__api_key_pool_dirty")
        )
        if param_key in dirty_pools:
            return []
        if self.api_key_tier == "Free":
            legacy_keys = parse_llm_api_keys(
                ";".join(
                    (
                        self.get_param_value("api_key"),
                        self.get_param_value("multiple_keys"),
                    )
                )
            )
            if legacy_keys:
                return legacy_keys
        return get_llm_api_key_pool(
            self.provider,
            self.api_key_tier,
            for_ocr=True,
        )

    @property
    def api_key(self) -> str:
        api_keys = self.active_api_keys
        return api_keys[0] if api_keys else ""

    @property
    def multiple_keys_list(self) -> List[str]:
        return self.active_api_keys

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def reasoning_effort(self) -> str:
        if self.params is None or "reasoning effort" not in self.params:
            return "default"
        effort = str(self.get_param_value("reasoning effort")).lower()
        if effort not in GEMINI_REASONING_EFFORT_OPTIONS:
            return "default"
        return effort

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override_model") or None

    @property
    def language(self) -> str:
        return self.get_param_value("language")

    @property
    def detail_level(self) -> str:
        return self.get_param_value("detail_level")

    @property
    def prompt(self) -> str:
        return self.get_param_value("prompt")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self) -> int:
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self) -> int:
        return int(self.get_param_value("max_response_tokens"))

    @property
    def request_delay(self) -> float:
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 1.0

    def _respect_delay(self):
        # This logic is identical to the one in LLM_API_Translator
        current_time = time.time()
        rpm = self.requests_per_minute
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f}s."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f}s.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        # This logic is identical to the one in LLM_API_Translator
        rpm = self.requests_per_minute
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) for key {key[:6]}... reached. Waiting {wait_time:.2f}s."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.active_api_keys
        if not api_keys:
            self.logger.error("No API keys provided.")
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All API keys are rate-limited.")
        return None

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        api_key_to_use = self._select_api_key()
        
        if not api_key_to_use:
            if self.provider in ["LLM Studio", "Ollama"]:
                api_key_to_use = "dummy-key"
            else:
                return "[ERROR: No available API key]"

        if self.provider == "LLM Studio" and not self.endpoint:
            return "[ERROR: LLM Studio endpoint is required]"

        # Re-initialize client if key is different from the last one used
        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)

        self._respect_delay()
        try:
            lang_name = self.language
            prompt_text = (prompt_override or self.prompt).format(language=lang_name)

            image_content_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }

            if self.provider in ["OpenAI", "Google", "Grok", "OpenRouter"]:
                detail_setting = self.detail_level
                if detail_setting in ["low", "high"]:
                    image_content_part["image_url"]["detail"] = detail_setting

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        image_content_part,
                    ],
                }
            ]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            model_name = self.override_model or self.model
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]

            self.logger.debug(f"OCR request with model: {model_name}")

            api_args = {
                "model": model_name,
                "messages": messages,
                "max_tokens": self.max_response_tokens,
            }
            apply_google_reasoning_effort(
                api_args,
                self.provider,
                self.reasoning_effort,
            )
            response = self.client.chat.completions.create(**api_args)

            if response.choices and response.choices[0].message.content:
                full_text = (
                    response.choices[0].message.content.replace("\n", " ").strip()
                )
                self.logger.debug(f"OCR result: {full_text}")
                return full_text
            else:
                self.logger.warning("No text found in OCR response.")
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return f"[ERROR: {type(e).__name__}]"

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", cropped_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                blk.text = self.ocr(img_base64, prompt_override=kwargs.get("prompt"))
            else:
                blk.text = ""

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in {"free_api_keys", "paid_api_keys"}:
            dirty_pools = parse_llm_api_keys(
                self.get_param_value("__api_key_pool_dirty")
            )
            if param_key not in dirty_pools:
                dirty_pools.append(param_key)
                self.set_param_value(
                    "__api_key_pool_dirty",
                    ";".join(dirty_pools),
                )
        pool_params = [
            "api_key_tier",
            "free_api_keys",
            "paid_api_keys",
            "api_key",
            "multiple_keys",
            "provider",
        ]
        if param_key in ["endpoint", "proxy", *pool_params]:
            self.client = None  # Force re-initialization on next call
        if param_key in pool_params:
            self.current_key_index = 0
        if param_key in ["requests_per_minute", "delay"]:
            self.request_count_minute = 0
            self.minute_start_time = time.time()
            self.last_request_time = 0


def _build_fixed_provider_params(
    description: str,
    model_options: List[str],
    default_model: str,
    include_reasoning_effort: bool = False,
) -> Dict:
    params = deepcopy(LLM_OCR.params)
    params.pop("provider", None)
    params["model"]["options"] = model_options
    params["model"]["value"] = default_model
    if include_reasoning_effort:
        params["reasoning effort"] = {
            "type": "selector",
            "options": list(GEMINI_REASONING_EFFORT_OPTIONS),
            "value": "default",
            "description": (
                "Controls Gemini reasoning depth. "
                "Default uses the model's native setting."
            ),
        }
    params["description"] = description
    return params


LLM_OCR_DEPENDENCIES = ['openai>=2.8.1', 'httpx[socks,brotli]', 'pydantic']


class _FixedProviderLLMOCR(LLM_OCR):
    fixed_provider: str = ""
    params: Dict = {}

    @property
    def provider(self) -> str:
        return self.fixed_provider


@register_OCR("LLM OCR OpenAI")
class OpenAILLMOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "OpenAI"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
    )


@register_OCR("LLM OCR Google")
class GoogleLLMOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "Google"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
        True,
    )


@register_OCR("LLM OCR Grok")
class GrokLLMOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "Grok"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
    )


@register_OCR("LLM OCR OpenRouter")
class OpenRouterLLMOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "OpenRouter"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
    )


@register_OCR("LLM OCR Studio")
class LLMStudioOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "LLM Studio"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
    )


@register_OCR("LLM OCR Ollama")
class OllamaLLMOCR(_FixedProviderLLMOCR):
    dependencies = LLM_OCR_DEPENDENCIES
    fixed_provider = "Ollama"
    params = _build_fixed_provider_params(
        LLM_OCR_PROVIDER_DESCRIPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_MODEL_OPTIONS[fixed_provider],
        LLM_OCR_PROVIDER_DEFAULT_MODELS[fixed_provider],
    )
