import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from utils.env import (
    get_llm_single_api_key,
    load_dotenv,
    parse_dotenv,
    persist_llm_api_keys_from_config,
    sanitize_llm_api_keys,
)
from utils.config import ProgramConfig
from modules.ocr.ocr_llm_api import (
    GoogleLLMOCR,
    LLM_OCR_PROVIDER_MODEL_OPTIONS,
    OpenAILLMOCR,
)
from modules.translators.trans_llm_api_json import LLM_PROVIDER_MODEL_OPTIONS
from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    OpenAILLMTranslator,
)


class DotenvTest(unittest.TestCase):
    def test_load_dotenv_preserves_existing_environment_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = os.path.join(temp_dir, ".env")
            with open(dotenv_path, "w", encoding="utf8") as f:
                f.write('EXISTING=from-file\nQUOTED="hello world"\n')

            with patch.dict(os.environ, {"EXISTING": "from-env"}, clear=True):
                self.assertTrue(load_dotenv(dotenv_path))
                self.assertEqual(os.environ["EXISTING"], "from-env")
                self.assertEqual(os.environ["QUOTED"], "hello world")

    def test_persist_llm_keys_to_dotenv_and_sanitize_config_output(self):
        module_cfg = {
            "translator_params": {
                "LLM OpenAI": {
                    "apikey": "openai-test-key",
                    "multiple_keys": "openai-key-a;openai-key-b",
                },
                "LLM Google": {
                    "apikey": ".",
                    "multiple_keys": "",
                },
                "LLM Studio": {
                    "apikey": "local-placeholder",
                    "multiple_keys": "",
                },
            },
            "ocr_params": {
                "LLM OCR Google": {
                    "api_key": "google-ocr-test-key",
                    "multiple_keys": "",
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = os.path.join(temp_dir, ".env")
            self.assertTrue(persist_llm_api_keys_from_config(module_cfg, dotenv_path))
            dotenv_values = parse_dotenv(dotenv_path)

        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OPENAI_API_KEY"],
            "openai-test-key",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OPENAI_API_KEYS"],
            "openai-key-a;openai-key-b",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OCR_GOOGLE_API_KEY"],
            "google-ocr-test-key",
        )
        self.assertNotIn("BALLOONTRANS_LLM_GOOGLE_API_KEY", dotenv_values)
        self.assertNotIn("BALLOONTRANS_LLM_LLM_STUDIO_API_KEY", dotenv_values)

        sanitized = sanitize_llm_api_keys(copy.deepcopy(module_cfg))
        self.assertEqual(sanitized["translator_params"]["LLM OpenAI"]["apikey"], "")
        self.assertEqual(sanitized["translator_params"]["LLM OpenAI"]["multiple_keys"], "")
        self.assertEqual(sanitized["ocr_params"]["LLM OCR Google"]["api_key"], "")

    def test_llm_ocr_uses_translator_model_catalog(self):
        self.assertEqual(
            LLM_OCR_PROVIDER_MODEL_OPTIONS["OpenAI"],
            LLM_PROVIDER_MODEL_OPTIONS["OpenAI"],
        )
        self.assertEqual(
            OpenAILLMOCR.params["model"]["options"],
            LLM_PROVIDER_MODEL_OPTIONS["OpenAI"],
        )
        self.assertEqual(
            GoogleLLMOCR.params["model"]["options"],
            LLM_PROVIDER_MODEL_OPTIONS["Google"],
        )

    def test_gemini_flash_lite_model_rename_is_shared(self):
        old_model = "GGL: gemini-3.1-flash-lite-preview"
        new_model = "GGL: gemini-3.1-flash-lite"

        self.assertIn(new_model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertIn(new_model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertNotIn(old_model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertNotIn(old_model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])

    def test_gemini_35_flash_lite_model_is_shared(self):
        model = "GGL: gemini-3.5-flash-lite"

        self.assertIn(model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertIn(model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])

    def test_reasoning_effort_is_exposed_only_by_google_presets(self):
        expected = ["default", "minimal", "low", "medium", "high"]

        self.assertEqual(
            GoogleLLMTranslator.params["reasoning effort"]["options"], expected
        )
        self.assertEqual(
            GoogleLLMOCR.params["reasoning effort"]["options"], expected
        )
        self.assertEqual(
            GoogleLLMTranslator.params["reasoning effort"]["value"], "default"
        )
        self.assertEqual(
            GoogleLLMOCR.params["reasoning effort"]["value"], "default"
        )
        self.assertNotIn("reasoning effort", OpenAILLMTranslator.params)
        self.assertNotIn("reasoning effort", OpenAILLMOCR.params)

    def test_provider_specific_env_wins_over_standard_fallback(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-standard-key",
                "BALLOONTRANS_LLM_OPENAI_API_KEY": "openai-project-key",
            },
            clear=True,
        ):
            self.assertEqual(
                get_llm_single_api_key("OpenAI"),
                "openai-project-key",
            )

    def test_legacy_llm_ocr_config_is_migrated_to_fixed_provider(self):
        config_dict = {
            "module": {
                "ocr": "llm_ocr",
                "translator": "google",
                "translator_params": {
                    "LLM Google": {
                        "model": "GGL: gemini-3.1-flash-lite-preview",
                        "override model": "",
                    },
                },
                "ocr_params": {
                    "llm_ocr": {
                        "provider": "OpenAI",
                        "api_key": "openai-ocr-test-key",
                        "multiple_keys": "",
                        "model": "OAI: gpt-4o-mini",
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w", encoding="utf8") as f:
                json.dump(config_dict, f)

            config = ProgramConfig.load(config_path)

        self.assertEqual(config.module.ocr, "LLM OCR OpenAI")
        self.assertNotIn("llm_ocr", config.module.ocr_params)
        self.assertIn("LLM OCR OpenAI", config.module.ocr_params)
        self.assertNotIn("provider", config.module.ocr_params["LLM OCR OpenAI"])
        self.assertEqual(
            config.module.ocr_params["LLM OCR OpenAI"]["model"],
            "OAI: gpt-5.2",
        )
        self.assertEqual(
            config.module.translator_params["LLM Google"]["model"],
            "GGL: gemini-3.1-flash-lite",
        )

    def test_standard_provider_env_is_supported(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-standard-key"}, clear=True):
            self.assertEqual(
                get_llm_single_api_key("OpenRouter"),
                "openrouter-standard-key",
            )
