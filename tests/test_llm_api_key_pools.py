import os
import unittest
from unittest.mock import patch

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
        self.assertEqual(params["free_api_keys"]["type"], "editor")
        self.assertEqual(params["paid_api_keys"]["type"], "editor")

    def test_translator_rotates_only_selected_pool_and_resets_on_switch(self):
        translator = self.make_translator()

        self.assertEqual(
            [translator._select_api_key() for _ in range(3)],
            ["free-a", "free-b", "free-a"],
        )

        translator.updateParam("api_key_tier", "Paid")

        self.assertEqual(translator.current_key_index, 0)
        self.assertIsNone(translator.client)
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


if __name__ == "__main__":
    unittest.main()
