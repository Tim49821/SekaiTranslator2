import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from modules.context.adapter import LLMContextAdapterMixin
from modules.context.glossary import GlossaryEntry
from modules.context.history import (
    ContextAction,
    ContextReason,
    HistoryPage,
    RenderedHistoryPage,
)
from modules.context.params import build_llm_context_params
from modules.translators.base import BaseTranslator
from utils.config import RunStatus
from utils.textblock import TextBlock


class FakeContextProject:
    def __init__(self):
        self.load_identity = object()
        first = TextBlock(text=["Hero"])
        first.translation = "용사"
        second = TextBlock(text=["Mage"])
        second.translation = "마법사"
        self.pages = OrderedDict((
            ("001.png", [first]),
            ("002.png", [second]),
            ("003.png", [TextBlock(text=["Current"])]),
        ))
        self._image_info = {
            "001.png": {
                "finish_code": RunStatus.FIN_TRANSLATE,
                "translation_target": "한국어",
            },
            "002.png": {
                "finish_code": RunStatus.FIN_TRANSLATE,
                "translation_target": "한국어",
            },
            "003.png": {"finish_code": 0},
        }


class FakeContextTranslator(LLMContextAdapterMixin, BaseTranslator):
    concate_text = False
    params = build_llm_context_params()

    def _setup_translator(self):
        self.lang_map["日本語"] = "Japanese"
        self.lang_map["한국어"] = "Korean"
        self.lang_map["English"] = "English"
        self._history_window = None
        self.model_name = "demo-model"
        self.prompt_signature = "demo-prompt"
        self.translate_call = None

    def _translate(self, src_list, **kwargs):
        self.translate_call = (tuple(src_list), kwargs)
        return list(src_list)

    def _context_model_name(self):
        return self.model_name

    def _context_prompt_signature(self):
        return self.prompt_signature

    def _render_history_page(self, page):
        messages = (
            ("user", "|".join(page.sources)),
            ("assistant", "|".join(page.translations)),
        )
        return RenderedHistoryPage(page, messages, 3)


class LLMContextAdapterTest(unittest.TestCase):
    def setUp(self):
        FakeContextTranslator.params = build_llm_context_params()

    def make_translator(self, **params):
        return FakeContextTranslator("日本語", "한국어", **params)

    def committed_translator(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 10}
        )
        project = FakeContextProject()
        context = translator._snapshot_request_context(project, "003.png")
        translator._commit_request_context(context)
        return translator, project, context

    def test_parameter_schema_is_deeply_fresh_and_uses_existing_file_picker(self):
        first = build_llm_context_params()
        second = build_llm_context_params()

        self.assertIsNot(first, second)
        for key in first:
            self.assertIsNot(first[key], second[key])
        self.assertIsNot(first["context mode"]["options"], second["context mode"]["options"])
        self.assertIsNot(first["glossary path"]["options"], second["glossary path"]["options"])
        self.assertIsNot(first["glossary mode"]["options"], second["glossary mode"]["options"])
        self.assertEqual(first["context mode"]["options"], ["page", "history"])
        self.assertEqual(first["history token budget"]["value"], 4096)
        self.assertTrue(first["glossary path"]["editable"])
        self.assertTrue(first["glossary path"]["path_selector"])
        self.assertEqual(first["glossary mode"]["options"], ["matching", "all"])

    def test_history_snapshot_uses_only_complete_past_target_pages(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 10}
        )

        context = translator._snapshot_request_context(
            FakeContextProject(), "003.png"
        )

        self.assertEqual(
            [page.page_key for page in context.history],
            ["001.png", "002.png"],
        )
        self.assertEqual(context.glossary, ())
        self.assertEqual(context.diagnostic.action, ContextAction.REBUILD)

    def test_page_mode_without_glossary_returns_none_and_clears_window(self):
        translator = self.make_translator()
        translator._history_window = object()

        self.assertIsNone(
            translator._snapshot_request_context(FakeContextProject(), "003.png")
        )
        self.assertIsNone(translator._history_window)

    def test_history_snapshot_rejects_target_empty_and_error_translations(self):
        cases = (
            ("unfinished", lambda project: project._image_info["001.png"].update(
                finish_code=0
            )),
            ("target", lambda project: project._image_info["001.png"].update(
                translation_target="English"
            )),
            ("empty", lambda project: setattr(
                project.pages["001.png"][0], "translation", ""
            )),
            ("error", lambda project: setattr(
                project.pages["001.png"][0],
                "translation",
                "[ERROR: API Failed]",
            )),
        )
        translator = self.make_translator()

        for label, mutate in cases:
            with self.subTest(label=label):
                project = FakeContextProject()
                mutate(project)
                self.assertIsNone(
                    translator._snapshot_history_page(project, "001.png")
                )

    def test_history_snapshot_accepts_legacy_target_and_aligns_nonempty_sources(self):
        translator = self.make_translator()
        project = FakeContextProject()
        project._image_info["001.png"].pop("translation_target")
        empty = TextBlock(text=[""])
        empty.translation = "ignored"
        project.pages["001.png"].append(empty)

        snapshot = translator._snapshot_history_page(project, "001.png")

        self.assertEqual(
            snapshot,
            HistoryPage("001.png", ("Hero",), ("용사",)),
        )
        self.assertTrue(all(isinstance(value, str) for value in snapshot.sources))
        self.assertTrue(all(isinstance(value, str) for value in snapshot.translations))

    def test_glossary_snapshot_is_frozen_and_matching_selects_current_terms(self):
        translator = self.make_translator()
        project = FakeContextProject()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.json"
            path.write_text(
                '[{"src":"Hero","dst":"용사"},{"src":"Mage","dst":"마법사"}]',
                encoding="utf-8",
            )
            translator.updateParam("glossary path", str(path))

            context = translator._snapshot_request_context(project, "003.png")
            path.write_text(
                '[{"src":"Hero","dst":"영웅"}]',
                encoding="utf-8",
            )

        self.assertEqual(
            context.glossary,
            (
                GlossaryEntry("Hero", "용사"),
                GlossaryEntry("Mage", "마법사"),
            ),
        )
        self.assertEqual(
            translator._selected_glossary(context, ["The HERO returns"]),
            (GlossaryEntry("Hero", "용사"),),
        )

    def test_all_glossary_mode_returns_every_frozen_entry(self):
        translator = self.make_translator()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.tsv"
            path.write_text("Hero\t용사\nMage\t마법사\n", encoding="utf-8")
            translator.updateParam("glossary path", str(path))
            translator.updateParam("glossary mode", "all")
            context = translator._snapshot_request_context(
                FakeContextProject(), "003.png"
            )

        self.assertEqual(
            translator._selected_glossary(context, ["No terms here"]),
            context.glossary,
        )

    def test_committed_window_rebuilds_for_model_prompt_and_language_changes(self):
        changes = (
            ("model", lambda translator, project: setattr(
                translator, "model_name", "other-model"
            )),
            ("prompt", lambda translator, project: setattr(
                translator, "prompt_signature", "other-prompt"
            )),
            ("source", lambda translator, project: translator.set_source("English")),
            ("target", self._change_target_to_english),
        )

        for label, change in changes:
            with self.subTest(label=label):
                translator, project, original = self.committed_translator()
                change(translator, project)

                rebuilt = translator._snapshot_request_context(project, "003.png")

                self.assertNotEqual(rebuilt.window_key, original.window_key)
                self.assertEqual(
                    rebuilt.diagnostic.rebuild_reason,
                    ContextReason.SETTINGS_CHANGED,
                )

    def test_update_param_clears_context_settings_but_not_unrelated_settings(self):
        for key, value in (
            ("context mode", "page"),
            ("history token budget", 20),
        ):
            with self.subTest(key=key):
                translator, _project, _context = self.committed_translator()
                self.assertIsNotNone(translator._history_window)
                translator.updateParam(key, value)
                self.assertIsNone(translator._history_window)

        translator, _project, _context = self.committed_translator()
        translator.params["unrelated"] = {"value": "before", "data_type": str}
        translator.updateParam("unrelated", "after")
        self.assertIsNotNone(translator._history_window)
        self.assertEqual(translator.get_param_value("unrelated"), "after")

    def test_invalid_runtime_params_use_safe_defaults_without_rewriting_config(self):
        translator = self.make_translator()
        invalid_glossary_mode = object()
        translator.params["context mode"]["value"] = "history"
        translator.params["history token budget"]["value"] = True
        translator.params["glossary mode"]["value"] = invalid_glossary_mode
        translator.params["glossary path"]["value"] = []
        translator.params["unrelated"] = {"value": "keep", "data_type": str}

        context = translator._snapshot_request_context(
            FakeContextProject(), "003.png"
        )

        self.assertEqual(context.history_budget, 4096)
        self.assertEqual(context.glossary, ())
        self.assertEqual(translator.params["history token budget"]["value"], True)
        self.assertIs(
            translator.params["glossary mode"]["value"],
            invalid_glossary_mode,
        )
        self.assertEqual(translator.params["glossary path"]["value"], [])
        self.assertEqual(translator.params["unrelated"]["value"], "keep")

    def test_invalid_context_mode_normalizes_to_page_without_rewriting_config(self):
        translator = self.make_translator()
        invalid_context_mode = object()
        translator.params["context mode"]["value"] = invalid_context_mode
        translator._history_window = object()

        context = translator._snapshot_request_context(
            FakeContextProject(), "003.png"
        )

        self.assertIsNone(context)
        self.assertIsNone(translator._history_window)
        self.assertIs(
            translator.params["context mode"]["value"],
            invalid_context_mode,
        )

    def test_translate_hook_freezes_context_without_speculative_commit(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 10}
        )
        project = FakeContextProject()

        result = translator._translate_with_context(
            ["Current"],
            project=project,
            page_key="003.png",
            commit_history_window=True,
        )

        self.assertEqual(result, ["Current"])
        _sources, kwargs = translator.translate_call
        self.assertEqual(kwargs["page_key"], "003.png")
        self.assertTrue(kwargs["commit_history_window"])
        self.assertEqual(
            [page.page_key for page in kwargs["request_context"].history],
            ["001.png", "002.png"],
        )
        self.assertIsNone(translator._history_window)

    def test_unload_model_clears_committed_window(self):
        translator, _project, _context = self.committed_translator()

        unloaded = translator.unload_model()

        self.assertFalse(unloaded)
        self.assertIsNone(translator._history_window)

    @staticmethod
    def _change_target_to_english(translator, project):
        translator.set_target("English")
        project._image_info["001.png"]["translation_target"] = "English"
        project._image_info["002.png"]["translation_target"] = "English"


if __name__ == "__main__":
    unittest.main()
