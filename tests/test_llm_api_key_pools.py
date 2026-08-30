import os
import unittest
from unittest.mock import patch

from modules.ocr.ocr_llm_api import OpenAILLMOCR
from modules.translators.trans_llm_api_json import OpenAILLMTranslator


class LLMTranslatorKeyPoolTest(unittest.TestCase):
    def setUp(self):
        self.reset_key_pool_params()

    def tearDown(self):
        self.reset_key_pool_params()

    @staticmethod
    def reset_key_pool_params():
        params = OpenAILLMTranslator.params
        params["api_key_tier"]["value"] = "Free"
        params["free_api_keys"]["value"] = ""
        params["paid_api_keys"]["value"] = ""
        params["apikey"]["value"] = ""
        params["multiple_keys"]["value"] = ""
        params["__api_key_pool_dirty"]["value"] = ""

    def make_translator(self, **params):
        return OpenAILLMTranslator(
            "日本語",
            "한국어",
            raise_unsupported_lang=False,
            **{
                "api_key_tier": "Free",
                "free_api_keys": "free-a;free-b",
                "paid_api_keys": "paid-a;paid-b",
                "max requests per minute": 0,
                **params,
            },
        )

    def test_translator_schema_defaults_to_free_with_two_pool_editors(self):
        params = OpenAILLMTranslator.params

        self.assertEqual(params["api_key_tier"]["value"], "Free")
        self.assertEqual(params["api_key_tier"]["options"], ["Free", "Paid"])
        self.assertEqual(
            params["api_key_tier"]["display_name"],
            "API key tier",
        )
        self.assertEqual(params["free_api_keys"]["type"], "editor")
        self.assertEqual(
            params["free_api_keys"]["display_name"],
            "Free API keys",
        )
        self.assertEqual(params["paid_api_keys"]["type"], "editor")
        self.assertEqual(
            params["paid_api_keys"]["display_name"],
            "Paid API keys",
        )

    def test_translator_rotates_only_selected_pool_and_resets_on_switch(self):
        translator = self.make_translator()
        committed_window = object()
        translator._history_window = committed_window

        self.assertEqual(
            [translator._select_api_key() for _ in range(3)],
            ["free-a", "free-b", "free-a"],
        )

        translator.updateParam("api_key_tier", "Paid")

        self.assertEqual(translator.current_key_index, 0)
        self.assertIsNone(translator.client)
        self.assertIs(translator._history_window, committed_window)
        self.assertEqual(
            [translator._select_api_key() for _ in range(3)],
            ["paid-a", "paid-b", "paid-a"],
        )

    def test_translator_single_paid_key_uses_pool_rotation(self):
        translator = self.make_translator(
            api_key_tier="Paid",
            paid_api_keys="paid-only",
        )

        self.assertEqual(translator._select_api_key(), "paid-only")
        self.assertEqual(translator._select_api_key(), "paid-only")

    def test_translator_empty_paid_pool_never_falls_back_to_free(self):
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, {}, clear=True
        ):
            translator = self.make_translator(
                api_key_tier="Paid",
                paid_api_keys="",
            )

            self.assertIsNone(translator._select_api_key())

    def test_translator_cleared_editor_immediately_blocks_stored_pool(self):
        env = {"BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "stale-free"}
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            translator = self.make_translator(free_api_keys="replacement")
            self.assertEqual(translator._select_api_key(), "replacement")

            translator.updateParam("free_api_keys", "")

            self.assertIsNone(translator._select_api_key())

    def test_translator_legacy_params_are_combined_as_free_pool(self):
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, {}, clear=True
        ):
            translator = OpenAILLMTranslator(
                "日本語",
                "한국어",
                raise_unsupported_lang=False,
                **{
                    "apikey": "legacy-single",
                    "multiple_keys": (
                        "legacy-a;legacy-single;legacy-b"
                    ),
                    "max requests per minute": 0,
                },
            )

            self.assertEqual(
                [translator._select_api_key() for _ in range(4)],
                [
                    "legacy-single",
                    "legacy-a",
                    "legacy-b",
                    "legacy-single",
                ],
            )


class LLMOCRKeyPoolTest(unittest.TestCase):
    def setUp(self):
        self.reset_key_pool_params()

    def tearDown(self):
        self.reset_key_pool_params()

    @staticmethod
    def reset_key_pool_params():
        params = OpenAILLMOCR.params
        defaults = {
            "api_key_tier": "Free",
            "free_api_keys": "",
            "paid_api_keys": "",
            "api_key": "",
            "multiple_keys": "",
            "__api_key_pool_dirty": "",
        }
        for key, value in defaults.items():
            if key in params:
                params[key]["value"] = value

    def make_ocr(self, **params):
        return OpenAILLMOCR(
            **{
                "api_key_tier": "Free",
                "free_api_keys": "ocr-free-a;ocr-free-b",
                "paid_api_keys": "ocr-paid-a;ocr-paid-b",
                "requests_per_minute": 0,
                **params,
            }
        )

    def test_ocr_schema_defaults_to_free_with_two_pool_editors(self):
        params = OpenAILLMOCR.params

        self.assertEqual(params["api_key_tier"]["value"], "Free")
        self.assertEqual(params["api_key_tier"]["options"], ["Free", "Paid"])
        self.assertEqual(
            params["api_key_tier"]["display_name"],
            "API key tier",
        )
        self.assertEqual(params["free_api_keys"]["type"], "editor")
        self.assertEqual(
            params["free_api_keys"]["display_name"],
            "Free API keys",
        )
        self.assertEqual(params["paid_api_keys"]["type"], "editor")
        self.assertEqual(
            params["paid_api_keys"]["display_name"],
            "Paid API keys",
        )

    def test_ocr_rotates_only_selected_pool_and_resets_on_switch(self):
        ocr = self.make_ocr()

        self.assertEqual(
            [ocr._select_api_key() for _ in range(3)],
            ["ocr-free-a", "ocr-free-b", "ocr-free-a"],
        )

        ocr.updateParam("api_key_tier", "Paid")

        self.assertEqual(ocr.current_key_index, 0)
        self.assertIsNone(ocr.client)
        self.assertEqual(
            [ocr._select_api_key() for _ in range(3)],
            ["ocr-paid-a", "ocr-paid-b", "ocr-paid-a"],
        )

    def test_ocr_empty_paid_pool_never_falls_back_to_free(self):
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, {}, clear=True
        ):
            ocr = self.make_ocr(
                api_key_tier="Paid",
                paid_api_keys="",
            )

            self.assertIsNone(ocr._select_api_key())

    def test_ocr_cleared_editor_immediately_blocks_stored_pool(self):
        env = {"BALLOONTRANS_LLM_OCR_OPENAI_FREE_API_KEYS": "stale-free"}
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            ocr = self.make_ocr(free_api_keys="replacement")
            self.assertEqual(ocr._select_api_key(), "replacement")

            ocr.updateParam("free_api_keys", "")

            self.assertIsNone(ocr._select_api_key())

    def test_ocr_legacy_params_are_combined_as_free_pool(self):
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, {}, clear=True
        ):
            ocr = OpenAILLMOCR(
                **{
                    "api_key": "legacy-ocr-single",
                    "multiple_keys": (
                        "legacy-ocr-a;legacy-ocr-single;legacy-ocr-b"
                    ),
                    "requests_per_minute": 0,
                }
            )

            self.assertEqual(
                [ocr._select_api_key() for _ in range(4)],
                [
                    "legacy-ocr-single",
                    "legacy-ocr-a",
                    "legacy-ocr-b",
                    "legacy-ocr-single",
                ],
            )

    def test_ocr_environment_pool_is_independent_from_translator_pool(self):
        env = {
            "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "translator-free",
            "BALLOONTRANS_LLM_OCR_OPENAI_FREE_API_KEYS": "ocr-free",
        }
        with patch("utils.env.load_dotenv"), patch.dict(
            os.environ, env, clear=True
        ):
            ocr = OpenAILLMOCR(
                **{
                    "api_key_tier": "Free",
                    "free_api_keys": "",
                    "requests_per_minute": 0,
                }
            )

            self.assertEqual(ocr._select_api_key(), "ocr-free")


if __name__ == "__main__":
    unittest.main()
