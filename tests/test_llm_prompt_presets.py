import os
import unittest
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from modules.translators import trans_llm_api_json as llm_translator
from ui import module_parse_widgets


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


class PromptPresetManagerTest(unittest.TestCase):
    def setUp(self):
        self.app = ensure_app()
        self.params = deepcopy(llm_translator.OpenAILLMTranslator.params)
        self.manager_class = getattr(
            module_parse_widgets,
            "ParamPresetManager",
            None,
        )

    def make_manager(self):
        self.assertIsNotNone(self.manager_class)
        return self.manager_class(
            "system prompt presets",
            self.params["system prompt presets"],
            self.params,
        )

    def test_existing_custom_prompt_becomes_default(self):
        self.params["system_prompt"]["value"] = "My existing custom prompt"

        manager = self.make_manager()

        self.assertEqual(manager.presets["Default"], "My existing custom prompt")
        self.assertEqual(manager.editor.text(), "My existing custom prompt")
        manager.close()

    def test_reloading_nondefault_selection_preserves_default(self):
        state = self.params["system prompt presets"]["value"]
        original_default = state["styles"]["Default"]
        state["selected"] = "Japanese Doujin/Manga → Korean"
        self.params["system_prompt"]["value"] = state["styles"][state["selected"]]

        manager = self.make_manager()

        self.assertEqual(manager.presets["Default"], original_default)
        self.assertEqual(manager.selected, "Japanese Doujin/Manga → Korean")
        manager.close()

    def test_selecting_preset_updates_active_system_prompt(self):
        manager = self.make_manager()

        manager.on_selected_preset_changed("Japanese Doujin/Manga → Korean")

        self.assertEqual(
            self.params["system_prompt"]["value"],
            llm_translator.JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT,
        )
        self.assertEqual(
            manager.editor.text(),
            llm_translator.JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT,
        )
        manager.close()

    def test_invalid_saved_state_falls_back_to_builtins(self):
        self.params["system prompt presets"]["value"] = {
            "selected": "Missing",
            "styles": [],
        }

        manager = self.make_manager()

        self.assertEqual(
            list(manager.presets),
            ["Default", "Japanese Doujin/Manga → Korean"],
        )
        self.assertEqual(manager.selected, "Default")
        manager.close()

    def test_add_replace_delete_updates_state_and_protects_default(self):
        manager = self.make_manager()
        manager.editor.setText("First custom prompt")

        self.assertTrue(manager.add_preset("My Prompt"))
        manager.editor.setText("Replacement prompt")
        manager.replace_selected_preset()
        self.assertEqual(manager.presets["My Prompt"], "Replacement prompt")
        self.assertTrue(manager.delete_selected_preset())
        self.assertNotIn("My Prompt", manager.presets)
        self.assertEqual(manager.selected, "Default")
        self.assertFalse(manager.delete_selected_preset())
        manager.close()

    def test_param_widget_builds_system_prompt_manager(self):
        self.assertIsNotNone(self.manager_class)

        widget = module_parse_widgets.ParamWidget(self.params)
        managers = widget.findChildren(self.manager_class)

        self.assertEqual(len(managers), 1)
        self.assertEqual(managers[0].target_param, "system_prompt")
        widget.close()


if __name__ == "__main__":
    unittest.main()
