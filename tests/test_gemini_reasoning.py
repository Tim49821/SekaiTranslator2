import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.ocr.ocr_llm_api import GoogleLLMOCR
from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    apply_google_reasoning_effort,
)


class IsolatedGoogleLLMTranslator(GoogleLLMTranslator):
    params = deepcopy(GoogleLLMTranslator.params)


class IsolatedGoogleLLMOCR(GoogleLLMOCR):
    params = deepcopy(GoogleLLMOCR.params)


def translation_completion():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"translations":[{"id":1,"translation":"안녕"}]}'
                )
            )
        ],
        usage=None,
    )


def ocr_completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="こんにちは"))],
        usage=None,
    )


class GeminiReasoningArgumentTest(unittest.TestCase):
    def test_ignores_unsupported_reasoning_effort(self):
        api_args = {}

        apply_google_reasoning_effort(api_args, "Google", "unsupported")

        self.assertEqual(api_args, {})


class GeminiTranslatorReasoningTest(unittest.TestCase):
    def request_mock(
        self,
        effort,
        model="GGL: gemini-3.1-pro-preview",
        override_model="",
    ):
        translator = IsolatedGoogleLLMTranslator(
            "日本語",
            "한국어",
            raise_unsupported_lang=False,
            **{
                "apikey": "test-key",
                "model": model,
                "override model": override_model,
                "reasoning effort": effort,
                "delay": 0,
            },
        )
        create = MagicMock(return_value=translation_completion())
        translator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.object(
            translator, "_select_api_key", return_value="test-key"
        ):
            with patch.object(
                translator, "_initialize_client", return_value=True
            ):
                with patch.object(translator, "_respect_delay"):
                    translator._request_translation([
                        {"role": "user", "content": "translate this"},
                    ])
        return create

    def test_sends_selected_reasoning_effort(self):
        create = self.request_mock("high")

        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

    def test_omits_reasoning_effort_for_default(self):
        create = self.request_mock("default")

        self.assertNotIn("reasoning_effort", create.call_args.kwargs)

    def test_omits_invalid_reasoning_effort(self):
        create = self.request_mock("unsupported")

        self.assertNotIn("reasoning_effort", create.call_args.kwargs)

    def test_gemini_36_flash_omits_legacy_sampling_parameters(self):
        create = self.request_mock("high", model="GGL: gemini-3.6-flash")

        self.assertNotIn("temperature", create.call_args.kwargs)
        self.assertNotIn("top_p", create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

    def test_gemini_35_flash_lite_omits_legacy_sampling_parameters(self):
        create = self.request_mock("default", model="GGL: gemini-3.5-flash-lite")

        self.assertNotIn("temperature", create.call_args.kwargs)
        self.assertNotIn("top_p", create.call_args.kwargs)

    def test_legacy_gemini_model_keeps_sampling_parameters(self):
        create = self.request_mock("default", model="GGL: gemini-3.1-pro-preview")

        self.assertEqual(create.call_args.kwargs["temperature"], 0.1)
        self.assertEqual(create.call_args.kwargs["top_p"], 1.0)

    def test_modern_override_model_omits_legacy_sampling_parameters(self):
        create = self.request_mock(
            "default",
            model="GGL: gemini-3.1-pro-preview",
            override_model="gemini-3.6-flash",
        )

        self.assertEqual(create.call_args.kwargs["model"], "gemini-3.6-flash")
        self.assertNotIn("temperature", create.call_args.kwargs)
        self.assertNotIn("top_p", create.call_args.kwargs)


class GeminiOCRReasoningTest(unittest.TestCase):
    def request_mock(self, effort):
        ocr = IsolatedGoogleLLMOCR(
            **{
                "api_key": "test-key",
                "reasoning effort": effort,
                "delay": 0,
            }
        )
        create = MagicMock(return_value=ocr_completion())
        ocr.client = SimpleNamespace(
            api_key="test-key",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with patch.object(ocr, "_select_api_key", return_value="test-key"):
            with patch.object(ocr, "_respect_delay"):
                self.assertEqual(ocr.ocr("encoded-image"), "こんにちは")
        return create

    def test_sends_selected_reasoning_effort(self):
        create = self.request_mock("high")

        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

    def test_omits_reasoning_effort_for_default(self):
        create = self.request_mock("default")

        self.assertNotIn("reasoning_effort", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
