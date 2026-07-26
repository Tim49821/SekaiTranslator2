import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from utils.env import (
    get_llm_api_key_pool,
    get_llm_single_api_key,
    hydrate_llm_api_key_params_from_dotenv,
    load_dotenv,
    parse_llm_api_keys,
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
from modules.translators.trans_llm_api_json import (
    LLM_PROVIDER_DEFAULT_MODELS,
    LLM_PROVIDER_MODEL_OPTIONS,
)
from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    OpenAILLMTranslator,
)


class DotenvTest(unittest.TestCase):
    def test_tier_pools_are_independent_and_deduplicated(self):
        env = {
            "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "free-a; free-b\nfree-a",
            "BALLOONTRANS_LLM_OPENAI_PAID_API_KEYS": "paid-a;paid-b",
            "BALLOONTRANS_LLM_OCR_OPENAI_FREE_API_KEYS": "ocr-free",
        }

        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            self.assertEqual(
                get_llm_api_key_pool("OpenAI", "Free"),
                ["free-a", "free-b"],
            )
            self.assertEqual(
                get_llm_api_key_pool("OpenAI", "Paid"),
                ["paid-a", "paid-b"],
            )
            self.assertEqual(
                get_llm_api_key_pool("OpenAI", "Free", for_ocr=True),
                ["ocr-free"],
            )

    def test_legacy_keys_form_default_free_pool(self):
        env = {
            "BALLOONTRANS_LLM_GOOGLE_API_KEY": "legacy-single",
            "BALLOONTRANS_LLM_GOOGLE_API_KEYS": (
                "legacy-a;legacy-single;legacy-b"
            ),
        }

        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            self.assertEqual(
                get_llm_api_key_pool("Google", "unknown"),
                ["legacy-single", "legacy-a", "legacy-b"],
            )
            self.assertEqual(get_llm_api_key_pool("Google", "Paid"), [])

    def test_ocr_free_pool_does_not_read_translator_legacy_variables(self):
        env = {
            "BALLOONTRANS_LLM_OPENAI_API_KEY": "translator-single",
            "BALLOONTRANS_LLM_OPENAI_API_KEYS": "translator-a;translator-b",
        }

        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            self.assertEqual(
                get_llm_api_key_pool("OpenAI", "Free", for_ocr=True),
                [],
            )

    def test_explicit_empty_tier_pool_blocks_legacy_fallback(self):
        env = {
            "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "",
            "BALLOONTRANS_LLM_OPENAI_API_KEY": "legacy-single",
            "OPENAI_API_KEY": "standard-single",
        }

        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            self.assertEqual(get_llm_api_key_pool("OpenAI", "Free"), [])

    def test_key_parser_accepts_newlines_and_ignores_empty_entries(self):
        self.assertEqual(
            parse_llm_api_keys(" key-a\n\nkey-b ; key-a ; "),
            ["key-a", "key-b"],
        )

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
                "LLM Grok": {
                    "free_api_keys": "grok-free-a\ngrok-free-b",
                    "paid_api_keys": "grok-paid",
                },
                "LLM Studio": {
                    "apikey": "local-placeholder",
                    "multiple_keys": "",
                },
            },
            "ocr_params": {
                "LLM OCR Google": {
                    "api_key": "google-ocr-test-key",
                    "multiple_keys": "google-ocr-key-b",
                },
                "LLM OCR OpenAI": {
                    "free_api_keys": "ocr-free-a;ocr-free-b",
                    "paid_api_keys": "ocr-paid",
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
            dotenv_values["BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS"],
            "openai-test-key;openai-key-a;openai-key-b",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_GROK_FREE_API_KEYS"],
            "grok-free-a;grok-free-b",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_GROK_PAID_API_KEYS"],
            "grok-paid",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OCR_GOOGLE_API_KEY"],
            "google-ocr-test-key",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OCR_GOOGLE_FREE_API_KEYS"],
            "google-ocr-test-key;google-ocr-key-b",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OCR_OPENAI_FREE_API_KEYS"],
            "ocr-free-a;ocr-free-b",
        )
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OCR_OPENAI_PAID_API_KEYS"],
            "ocr-paid",
        )
        self.assertNotIn("BALLOONTRANS_LLM_GOOGLE_API_KEY", dotenv_values)
        self.assertNotIn("BALLOONTRANS_LLM_LLM_STUDIO_API_KEY", dotenv_values)

        sanitized = sanitize_llm_api_keys(copy.deepcopy(module_cfg))
        self.assertEqual(sanitized["translator_params"]["LLM OpenAI"]["apikey"], "")
        self.assertEqual(sanitized["translator_params"]["LLM OpenAI"]["multiple_keys"], "")
        self.assertEqual(
            sanitized["translator_params"]["LLM Grok"]["free_api_keys"],
            "",
        )
        self.assertEqual(
            sanitized["translator_params"]["LLM Grok"]["paid_api_keys"],
            "",
        )
        self.assertEqual(sanitized["ocr_params"]["LLM OCR Google"]["api_key"], "")
        self.assertEqual(
            sanitized["ocr_params"]["LLM OCR OpenAI"]["free_api_keys"],
            "",
        )
        self.assertEqual(
            sanitized["ocr_params"]["LLM OCR OpenAI"]["paid_api_keys"],
            "",
        )

    def test_explicitly_cleared_pool_replaces_stored_value_with_empty_override(self):
        module_cfg = {
            "translator_params": {
                "LLM OpenAI": {
                    "free_api_keys": "",
                    "paid_api_keys": "",
                    "__api_key_pool_dirty": "free_api_keys",
                },
            },
            "ocr_params": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = os.path.join(temp_dir, ".env")
            with open(dotenv_path, "w", encoding="utf8") as f:
                f.write(
                    "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS=stale-free\n"
                    "BALLOONTRANS_LLM_OPENAI_API_KEY=legacy-free\n"
                )

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(
                    persist_llm_api_keys_from_config(module_cfg, dotenv_path)
                )
                dotenv_values = parse_dotenv(dotenv_path)
                load_dotenv(dotenv_path, override=True)
                with patch("utils.env.load_dotenv"):
                    resolved_pool = get_llm_api_key_pool("OpenAI", "Free")

        self.assertIn("BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS", dotenv_values)
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS"],
            "",
        )
        self.assertEqual(resolved_pool, [])

    def test_dotenv_key_pools_hydrate_visible_editor_fields(self):
        module_params = {
            "LLM OpenAI": {
                "free_api_keys": {"value": ""},
                "paid_api_keys": {"value": ""},
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = os.path.join(temp_dir, ".env")
            with open(dotenv_path, "w", encoding="utf8") as f:
                f.write(
                    "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS=free-a;free-b\n"
                    "BALLOONTRANS_LLM_OPENAI_PAID_API_KEYS=paid-a\n"
                )

            hydrate_llm_api_key_params_from_dotenv(
                module_params,
                dotenv_path=dotenv_path,
            )

        self.assertEqual(
            module_params["LLM OpenAI"]["free_api_keys"]["value"],
            "free-a;free-b",
        )
        self.assertEqual(
            module_params["LLM OpenAI"]["paid_api_keys"]["value"],
            "paid-a",
        )

    def test_dotenv_hydration_does_not_shadow_external_environment_pool(self):
        module_params = {
            "LLM OpenAI": {
                "free_api_keys": {"value": ""},
                "paid_api_keys": {"value": ""},
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = os.path.join(temp_dir, ".env")
            with open(dotenv_path, "w", encoding="utf8") as f:
                f.write(
                    "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS=file-key\n"
                )

            with patch.dict(
                os.environ,
                {"BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "external-key"},
                clear=True,
            ):
                hydrate_llm_api_key_params_from_dotenv(
                    module_params,
                    dotenv_path=dotenv_path,
                )
                with patch("utils.env.load_dotenv"):
                    resolved_pool = get_llm_api_key_pool("OpenAI", "Free")

        self.assertEqual(
            module_params["LLM OpenAI"]["free_api_keys"]["value"],
            "",
        )
        self.assertEqual(resolved_pool, ["external-key"])

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

    def test_gemini_36_flash_model_is_shared(self):
        model = "GGL: gemini-3.6-flash"

        self.assertIn(model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertIn(model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])
        self.assertEqual(
            LLM_PROVIDER_DEFAULT_MODELS["Google"],
            "GGL: gemini-3.1-pro-preview",
        )

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

    def test_legacy_api_key_params_migrate_to_default_free_pools(self):
        config_dict = {
            "module": {
                "ocr": "LLM OCR Google",
                "translator": "LLM OpenAI",
                "translator_params": {
                    "LLM OpenAI": {
                        "apikey": "legacy-single",
                        "multiple_keys": (
                            "legacy-a;legacy-single;legacy-b"
                        ),
                    },
                },
                "ocr_params": {
                    "LLM OCR Google": {
                        "api_key": "legacy-ocr",
                        "multiple_keys": "legacy-ocr-b",
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            dotenv_path = os.path.join(temp_dir, ".env")
            with open(config_path, "w", encoding="utf8") as f:
                json.dump(config_dict, f)

            config = ProgramConfig.load(config_path)
            self.assertTrue(
                persist_llm_api_keys_from_config(
                    config.module,
                    dotenv_path,
                )
            )
            dotenv_values = parse_dotenv(dotenv_path)

        translator_params = config.module.translator_params["LLM OpenAI"]
        ocr_params = config.module.ocr_params["LLM OCR Google"]
        self.assertEqual(translator_params["api_key_tier"], "Free")
        self.assertEqual(
            translator_params["free_api_keys"],
            "legacy-single;legacy-a;legacy-b",
        )
        self.assertEqual(translator_params["paid_api_keys"], "")
        self.assertEqual(ocr_params["api_key_tier"], "Free")
        self.assertEqual(
            ocr_params["free_api_keys"],
            "legacy-ocr;legacy-ocr-b",
        )
        self.assertEqual(ocr_params["paid_api_keys"], "")
        self.assertEqual(
            dotenv_values["BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS"],
            "legacy-single;legacy-a;legacy-b",
        )
        self.assertEqual(
            dotenv_values[
                "BALLOONTRANS_LLM_OCR_GOOGLE_FREE_API_KEYS"
            ],
            "legacy-ocr;legacy-ocr-b",
        )

    def test_standard_provider_env_is_supported(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-standard-key"}, clear=True):
            self.assertEqual(
                get_llm_single_api_key("OpenRouter"),
                "openrouter-standard-key",
            )
