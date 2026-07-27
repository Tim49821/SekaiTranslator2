import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.translators import trans_llm_api_json as llm_translator


class LLMPromptPresetDefinitionTest(unittest.TestCase):
    provider_classes = (
        llm_translator.OpenAILLMTranslator,
        llm_translator.GoogleLLMTranslator,
        llm_translator.GrokLLMTranslator,
        llm_translator.OpenRouterLLMTranslator,
        llm_translator.LLMStudioTranslator,
    )

    def test_every_provider_has_independent_builtin_prompt_presets(self):
        for provider in self.provider_classes:
            self.assertIn("system prompt presets", provider.params)

        preset_values = [
            provider.params["system prompt presets"]["value"]
            for provider in self.provider_classes
        ]

        for provider in self.provider_classes:
            self.assertTrue(provider.params["system_prompt"]["hidden"])
            self.assertEqual(
                list(provider.params["system prompt presets"]["value"]["styles"]),
                ["Default", "Japanese Doujin/Manga → Korean"],
            )
        for index, value in enumerate(preset_values):
            for other in preset_values[index + 1:]:
                self.assertIsNot(value, other)
                self.assertIsNot(value["styles"], other["styles"])

    def test_manga_prompt_requires_contextual_faithful_korean_json(self):
        prompt = getattr(
            llm_translator,
            "JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT",
            "",
        )

        self.assertIn("Japanese", prompt)
        self.assertIn("natural Korean", prompt)
        self.assertIn("shared page context", prompt)
        self.assertIn("speech levels", prompt)
        self.assertIn("SFX", prompt)
        self.assertIn("vertical", prompt)
        self.assertIn("Do not censor", prompt)
        self.assertIn("exactly one", prompt)
        self.assertIn("numeric id", prompt)
        self.assertIn("valid JSON", prompt)


if __name__ == "__main__":
    unittest.main()
