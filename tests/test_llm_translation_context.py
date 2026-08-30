import json
import tempfile
import unittest
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from modules.context.adapter import LLMContextAdapterMixin
from modules.context.glossary import GlossaryEntry
from modules.context.errors import ContextLengthError
from modules.context.history import (
    ContextAction,
    ContextReason,
    HistoryPage,
    HistoryWindowKey,
    RenderedHistoryPage,
    RequestContext,
)
from modules.context.params import build_llm_context_params
from modules.translators.base import BaseTranslator
from modules.translators.trans_gemma4 import Gemma4E4BTranslator
from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    GrokLLMTranslator,
    LLMStudioTranslator,
    OpenAILLMTranslator,
    OpenRouterLLMTranslator,
    TranslationResponse,
)
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
        self.render_token_counts = {}
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
        return RenderedHistoryPage(
            page,
            messages,
            self.render_token_counts.get(page.page_key, 3),
        )


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

    def test_project_load_identity_change_rebuilds_committed_window(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 10}
        )
        project = FakeContextProject()
        initial = translator._snapshot_request_context(project, "002.png")
        translator._commit_request_context(initial)

        project.load_identity = object()
        rebuilt = translator._snapshot_request_context(project, "003.png")

        self.assertEqual(
            rebuilt.diagnostic.rebuild_reason,
            ContextReason.PROJECT_CHANGED,
        )
        self.assertEqual(rebuilt.diagnostic.action, ContextAction.REBUILD)
        self.assertIs(rebuilt.window_key.load_identity, project.load_identity)

    def test_changed_retained_snapshot_rebuilds_through_adapter(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 10}
        )
        project = FakeContextProject()
        initial = translator._snapshot_request_context(project, "002.png")
        translator._commit_request_context(initial)

        project.pages["001.png"][0].translation = "영웅"
        rebuilt = translator._snapshot_request_context(project, "003.png")

        self.assertEqual(
            rebuilt.diagnostic.rebuild_reason,
            ContextReason.SNAPSHOT_CHANGED,
        )
        self.assertEqual(rebuilt.diagnostic.action, ContextAction.REBUILD)
        first = next(
            page for page in rebuilt.history if page.page_key == "001.png"
        )
        self.assertEqual(first.snapshot.translations, ("영웅",))

    def test_adjacent_oversized_page_reuses_committed_window(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 7}
        )
        project = FakeContextProject()
        initial = translator._snapshot_request_context(project, "002.png")
        translator._commit_request_context(initial)
        translator.render_token_counts["002.png"] = 8

        reused = translator._snapshot_request_context(project, "003.png")

        self.assertEqual(
            [page.page_key for page in reused.history],
            ["001.png"],
        )
        self.assertEqual(reused.diagnostic.action, ContextAction.REUSE)
        self.assertEqual(
            reused.diagnostic.rebuild_reason,
            ContextReason.OVERSIZED_PAGE,
        )

    def test_adjacent_growth_keeps_pages_below_hard_budget_through_adapter(self):
        translator = self.make_translator(
            **{"context mode": "history", "history token budget": 7}
        )
        project = FakeContextProject()
        initial = translator._snapshot_request_context(project, "002.png")
        translator._commit_request_context(initial)

        grown = translator._snapshot_request_context(project, "003.png")

        self.assertEqual(
            [page.page_key for page in grown.history],
            ["001.png", "002.png"],
        )
        self.assertEqual(grown.diagnostic.action, ContextAction.GROW)
        self.assertEqual(grown.diagnostic.token_count, 6)
        self.assertEqual(grown.diagnostic.appended, 1)
        self.assertEqual(grown.diagnostic.evicted, 0)

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


class GemmaContextAdapterTest(unittest.TestCase):
    def make_translator(self):
        return Gemma4E4BTranslator(
            "日本語",
            "한국어",
            **{"worker python": "/fake/python"},
        )

    def request_context(self, translator):
        rendered = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        return RequestContext(
            history=(rendered,),
            glossary=(GlossaryEntry("Hero", "용사"),),
            glossary_mode="matching",
            history_budget=4096,
            window_key=HistoryWindowKey(object(), (("model", "gemma"),)),
            request_page_key="002.png",
        )

    @staticmethod
    def worker_result(*translations):
        return SimpleNamespace(
            stdout=json.dumps(
                {"translations": list(translations)},
                ensure_ascii=False,
            ),
            stderr="",
            returncode=0,
        )

    def translate_page(self, translator, context, completed, hooks=None):
        block = TextBlock(text=["Current"])
        hook_patch = patch.object(
            translator,
            "_postprocess_hooks",
            hooks if hooks is not None else OrderedDict(),
        )
        with patch.object(
            translator,
            "_snapshot_request_context",
            return_value=context,
        ), patch(
            "modules.translators.trans_gemma4.osp.isfile",
            return_value=True,
        ), patch(
            "modules.translators.trans_gemma4.subprocess.run",
            return_value=completed,
        ), hook_patch:
            success = translator.translate_textblk_lst(
                [block],
                project=object(),
                page_key="002.png",
                full_page=True,
            )
        return success, block

    def test_rendered_history_uses_plain_stable_glossary_free_messages(self):
        translator = self.make_translator()

        rendered = translator._render_history_page(
            HistoryPage("001.png", ("Hero", "Mage"), ("용사", "마법사"))
        )

        self.assertEqual(
            [role for role, _content in rendered.messages],
            ["user", "assistant"],
        )
        self.assertTrue(
            all(
                isinstance(role, str) and isinstance(content, str)
                for role, content in rendered.messages
            )
        )
        self.assertNotIn("glossary", rendered.messages[0][1].lower())
        self.assertEqual(
            rendered.messages[1][1],
            '{"translations":[{"id":1,"translation":"용사"},'
            '{"id":2,"translation":"마법사"}]}',
        )

    def test_successful_full_result_commits_only_after_postprocess(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before
        observed = []

        def observe_pending(translations, **_kwargs):
            observed.append((
                translator._history_window,
                translator._pending_request_context,
                list(translations),
            ))

        success, block = self.translate_page(
            translator,
            context,
            self.worker_result("현재"),
            OrderedDict((("observe_pending", observe_pending),)),
        )

        self.assertTrue(success)
        self.assertEqual(block.translation, "현재")
        self.assertEqual(observed, [(committed_before, context, ["현재"])])
        self.assertEqual(translator._history_window.request_page_key, "002.png")
        self.assertEqual(
            [page.page_key for page in translator._history_window.history],
            ["001.png"],
        )
        self.assertIsNone(translator._pending_request_context)

    def test_empty_and_error_results_clear_pending_without_commit(self):
        for translation in ("", "[ERROR: StructureError]"):
            with self.subTest(translation=translation):
                translator = self.make_translator()
                context = self.request_context(translator)
                committed_before = object()
                translator._history_window = committed_before

                success, block = self.translate_page(
                    translator,
                    context,
                    self.worker_result(translation),
                )

                self.assertFalse(success)
                self.assertEqual(block.translation, translation)
                self.assertIs(translator._history_window, committed_before)
                self.assertIsNone(translator._pending_request_context)

    def test_postprocess_truncation_clears_pending_without_commit(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before

        def truncate(translations, **_kwargs):
            translations.clear()

        success, block = self.translate_page(
            translator,
            context,
            self.worker_result("현재"),
            OrderedDict((("truncate", truncate),)),
        )

        self.assertFalse(success)
        self.assertEqual(block.translation, "")
        self.assertIs(translator._history_window, committed_before)
        self.assertIsNone(translator._pending_request_context)

    def test_postprocess_exception_clears_pending_and_propagates(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before
        error = RuntimeError("postprocess failed")

        def fail_postprocess(**_kwargs):
            raise error

        with self.assertRaises(RuntimeError) as raised:
            self.translate_page(
                translator,
                context,
                self.worker_result("현재"),
                OrderedDict((("fail", fail_postprocess),)),
            )

        self.assertIs(raised.exception, error)
        self.assertIs(translator._history_window, committed_before)
        self.assertIsNone(translator._pending_request_context)

    def test_wrong_length_worker_payload_logs_only_aggregate_metadata(self):
        translator = self.make_translator()
        translator.logger = MagicMock()
        secret = "SECRET-WORKER-TRANSLATION"

        with patch(
            "modules.translators.trans_gemma4.osp.isfile",
            return_value=True,
        ), patch(
            "modules.translators.trans_gemma4.subprocess.run",
            return_value=self.worker_result("현재", secret),
        ):
            result = translator._translate(["Current"])

        self.assertEqual(
            result,
            ["[ERROR: Gemma4 GGUF subprocess returned invalid translation payload.]"],
        )
        logged = repr(translator.logger.method_calls)
        self.assertIn("response_type", logged)
        self.assertIn("item_count", logged)
        self.assertNotIn(secret, logged)


class RemoteLLMContextTest(unittest.TestCase):
    provider_classes = (
        OpenAILLMTranslator,
        GoogleLLMTranslator,
        GrokLLMTranslator,
        OpenRouterLLMTranslator,
        LLMStudioTranslator,
    )

    def setUp(self):
        self.openai_params = deepcopy(OpenAILLMTranslator.params)

    def tearDown(self):
        OpenAILLMTranslator.params = self.openai_params

    def make_translator(self, **params):
        return OpenAILLMTranslator(
            "日本語",
            "한국어",
            raise_unsupported_lang=False,
            **{
                "free_api_keys": "test-key",
                "context mode": "page",
                "history token budget": 4096,
                "glossary path": "",
                "glossary mode": "matching",
                "max requests per minute": 0,
                "delay": 0,
                "retry attempts": 3,
                "retry timeout": 0,
                **params,
            },
        )

    @staticmethod
    def response(*translations):
        return TranslationResponse.model_validate({
            "translations": [
                {"id": index + 1, "translation": translation}
                for index, translation in enumerate(translations)
            ]
        })

    def request_context(self, translator):
        rendered = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        return RequestContext(
            history=(rendered,),
            history_budget=4096,
            window_key=HistoryWindowKey(
                object(),
                (("model", "remote"),),
            ),
            request_page_key="002.png",
        )

    def translate_page(self, translator, context, response, hooks=None):
        block = TextBlock(text=["Current"])
        hook_patch = patch.object(
            translator,
            "_postprocess_hooks",
            hooks if hooks is not None else OrderedDict(),
        )
        with patch.object(
            translator,
            "_snapshot_request_context",
            return_value=context,
        ), patch.object(
            translator,
            "_request_translation",
            return_value=response,
        ), hook_patch:
            success = translator.translate_textblk_lst(
                [block],
                project=object(),
                page_key="002.png",
                full_page=True,
            )
        return success, block

    def test_every_fixed_provider_gets_independent_context_params(self):
        values = [
            provider.params["glossary path"]
            for provider in self.provider_classes
        ]
        for provider in self.provider_classes:
            self.assertEqual(provider.params["context mode"]["value"], "page")
            self.assertEqual(
                provider.params["history token budget"]["value"],
                4096,
            )
        for index, value in enumerate(values):
            for other in values[index + 1:]:
                self.assertIsNot(value, other)

    def test_disabled_features_preserve_existing_two_message_shape(self):
        translator = self.make_translator()

        messages, prompt = translator._assemble_request(["こんにちは"])

        self.assertEqual(messages, [
            {"role": "system", "content": translator.system_prompt},
            {"role": "user", "content": prompt},
        ])
        self.assertEqual(
            prompt,
            "Please translate the following text snippets from Japanese to Korean. "
            "The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            "INPUT:\n[\n"
            "  {\n"
            '    "id": 1,\n'
            '    "source": "こんにちは"\n'
            "  }\n"
            "]",
        )

    def test_matching_glossary_is_only_in_current_user_message(self):
        translator = self.make_translator()
        page = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        context = RequestContext(
            history=(page,),
            glossary=(
                GlossaryEntry("Hero", "용사"),
                GlossaryEntry("Mage", "마법사"),
            ),
            glossary_mode="matching",
            history_budget=4096,
        )

        messages, _prompt = translator._assemble_request(
            ["Mage appears"],
            context,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertNotIn("glossary", messages[1]["content"].casefold())
        self.assertIn('"source":"Mage"', messages[-1]["content"])
        self.assertNotIn('"source":"Hero"', messages[-1]["content"])

    def test_all_glossary_is_stable_system_message_before_history(self):
        translator = self.make_translator()
        context = RequestContext(
            history=(),
            glossary=(GlossaryEntry("Hero", "용사"),),
            glossary_mode="all",
            history_budget=4096,
        )

        messages, _prompt = translator._assemble_request(
            ["Nothing matches"],
            context,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user"],
        )
        self.assertIn('"source":"Hero"', messages[1]["content"])
        self.assertNotIn("GLOSSARY", messages[-1]["content"])

    def test_successful_result_commits_only_after_postprocess(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before
        observed = []

        def observe_pending(translations, **_kwargs):
            observed.append((
                translator._history_window,
                translator._pending_request_context,
                list(translations),
            ))

        success, block = self.translate_page(
            translator,
            context,
            self.response("현재"),
            OrderedDict((("observe_pending", observe_pending),)),
        )

        self.assertTrue(success)
        self.assertEqual(block.translation, "현재")
        self.assertEqual(observed, [(committed_before, context, ["현재"])])
        self.assertEqual(translator._history_window.request_page_key, "002.png")
        self.assertIsNone(translator._pending_request_context)

    def test_empty_and_error_marker_results_clear_pending_without_commit(self):
        for translation in ("", "[ERROR: Provider Returned Error]"):
            with self.subTest(translation=translation):
                translator = self.make_translator()
                context = self.request_context(translator)
                committed_before = object()
                translator._history_window = committed_before

                success, block = self.translate_page(
                    translator,
                    context,
                    self.response(translation),
                )

                self.assertFalse(success)
                self.assertEqual(block.translation, translation)
                self.assertIs(translator._history_window, committed_before)
                self.assertIsNone(translator._pending_request_context)

    def test_postprocess_truncation_clears_pending_without_commit(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before

        def truncate(translations, **_kwargs):
            translations.clear()

        success, block = self.translate_page(
            translator,
            context,
            self.response("현재"),
            OrderedDict((("truncate", truncate),)),
        )

        self.assertFalse(success)
        self.assertEqual(block.translation, "")
        self.assertIs(translator._history_window, committed_before)
        self.assertIsNone(translator._pending_request_context)

    def test_postprocess_exception_clears_pending_and_propagates(self):
        translator = self.make_translator()
        context = self.request_context(translator)
        committed_before = object()
        translator._history_window = committed_before
        error = RuntimeError("postprocess failed")

        def fail_postprocess(**_kwargs):
            raise error

        with self.assertRaises(RuntimeError) as raised:
            self.translate_page(
                translator,
                context,
                self.response("현재"),
                OrderedDict((("fail", fail_postprocess),)),
            )

        self.assertIs(raised.exception, error)
        self.assertIs(translator._history_window, committed_before)
        self.assertIsNone(translator._pending_request_context)

    def test_ordinary_retry_reuses_equal_immutable_messages(self):
        translator = self.make_translator()
        requests = []

        def request(messages, **_kwargs):
            requests.append(deepcopy(messages))
            if len(requests) == 1:
                raise httpx.RequestError("temporary connection failure")
            return self.response("안녕하세요")

        with patch.object(
            translator,
            "_request_translation",
            side_effect=request,
        ):
            result = translator._translate(["こんにちは"])

        self.assertEqual(result, ["안녕하세요"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], requests[1])

    def test_context_recovery_evicts_whole_page_without_retry_or_early_commit(self):
        translator = self.make_translator(**{"retry attempts": 1})
        first = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        second = translator._render_history_page(
            HistoryPage("002.png", ("Mage",), ("마법사",))
        )
        context = RequestContext(
            history=(first, second),
            glossary=(GlossaryEntry("Current", "현재"),),
            glossary_mode="matching",
            history_budget=4096,
            window_key=HistoryWindowKey(object(), (("model", "demo"),)),
            request_page_key="003.png",
        )
        committed_before = object()
        translator._history_window = committed_before
        requests = []
        windows_during_requests = []

        def request(messages, **_kwargs):
            requests.append(deepcopy(messages))
            windows_during_requests.append(translator._history_window)
            if len(requests) == 1:
                raise ContextLengthError("maximum context length exceeded")
            return self.response("현재")

        with patch.object(
            translator,
            "_snapshot_request_context",
            return_value=context,
        ), patch.object(
            translator,
            "_request_translation",
            side_effect=request,
        ), patch.object(
            translator,
            "_postprocess_hooks",
            OrderedDict(),
        ):
            block = TextBlock(text=["Current"])
            success = translator.translate_textblk_lst(
                [block],
                project=object(),
                page_key="003.png",
                full_page=True,
            )

        self.assertTrue(success)
        self.assertEqual(block.translation, "현재")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1], [requests[0][0]] + requests[0][3:])
        self.assertEqual(requests[0][-1], requests[1][-1])
        self.assertIn('"source":"Current"', requests[1][-1]["content"])
        self.assertTrue(
            all(window is committed_before for window in windows_during_requests)
        )
        self.assertIsNot(translator._history_window, committed_before)
        self.assertEqual(
            [page.page_key for page in translator._history_window.history],
            ["002.png"],
        )
        self.assertIsNone(translator._pending_request_context)

    def test_context_recovery_final_failure_preserves_committed_window(self):
        translator = self.make_translator()
        page = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        context = RequestContext(
            history=(page,),
            history_budget=4096,
            window_key=HistoryWindowKey(object(), (("model", "demo"),)),
            request_page_key="002.png",
        )
        committed_before = object()
        translator._history_window = committed_before

        with patch.object(
            translator,
            "_snapshot_request_context",
            return_value=context,
        ), patch.object(
            translator,
            "_request_translation",
            side_effect=ContextLengthError("maximum context length exceeded"),
        ) as request, patch.object(
            translator,
            "_postprocess_hooks",
            OrderedDict(),
        ):
            block = TextBlock(text=["Current"])
            success = translator.translate_textblk_lst(
                [block],
                project=object(),
                page_key="002.png",
                full_page=True,
            )

        self.assertFalse(success)
        self.assertEqual(block.translation, "[ERROR: ContextLengthError]")
        self.assertEqual(request.call_count, 2)
        self.assertIs(translator._history_window, committed_before)
        self.assertIsNone(translator._pending_request_context)

    def test_request_passes_messages_untouched_and_logs_only_aggregate_usage(self):
        translator = self.make_translator()
        messages = [
            {"role": "system", "content": "private system"},
            {"role": "user", "content": "private glossary and source"},
        ]
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    '{"translations":[{"id":1,'
                    '"translation":"안녕하세요"}]}'
                )
            ))],
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=4,
                total_tokens=9,
            ),
        )
        create = MagicMock(return_value=completion)
        translator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        translator.logger = MagicMock()

        with patch.object(
            translator,
            "_select_api_key",
            return_value="test-key",
        ), patch.object(
            translator,
            "_initialize_client",
            return_value=True,
        ), patch.object(translator, "_respect_delay"):
            response = translator._request_translation(
                messages,
                usage_page_key="003.png\nsecret",
                usage_attempt=2,
            )

        self.assertEqual(response.translations[0].translation, "안녕하세요")
        self.assertIs(create.call_args.kwargs["messages"], messages)
        self.assertEqual(translator.token_count, 9)
        self.assertEqual(translator.token_count_last, 9)
        translator.logger.debug.assert_any_call(
            "LLM token usage: page=%s, attempt=%s, %s",
            "003.png secret",
            2,
            "prompt=5, completion=4, total=9",
        )
        logged = repr(translator.logger.method_calls)
        self.assertNotIn("private system", logged)
        self.assertNotIn("private glossary and source", logged)

    def test_request_classifies_context_errors_without_logging_provider_body(self):
        translator = self.make_translator()
        provider_secret = "private provider response"
        provider_error = RuntimeError(provider_secret)
        provider_error.status_code = 400
        provider_error.code = "context_length_exceeded"
        create = MagicMock(side_effect=provider_error)
        translator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        translator.logger = MagicMock()

        with patch.object(
            translator,
            "_select_api_key",
            return_value="test-key",
        ), patch.object(
            translator,
            "_initialize_client",
            return_value=True,
        ), patch.object(translator, "_respect_delay"):
            with self.assertRaises(ContextLengthError):
                translator._request_translation([])

        self.assertNotIn(
            provider_secret,
            repr(translator.logger.method_calls),
        )

    def test_validation_failure_logs_metadata_without_response_body(self):
        translator = self.make_translator()
        response_secret = "private malformed response body"
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=response_secret
            ))],
            usage=None,
        )
        create = MagicMock(return_value=completion)
        translator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        translator.logger = MagicMock()

        with patch.object(
            translator,
            "_select_api_key",
            return_value="test-key",
        ), patch.object(
            translator,
            "_initialize_client",
            return_value=True,
        ), patch.object(translator, "_respect_delay"):
            with self.assertRaises(json.JSONDecodeError):
                translator._request_translation([])

        logged = repr(translator.logger.method_calls)
        self.assertIn("response_chars", logged)
        self.assertNotIn(response_secret, logged)

    def test_remote_update_param_clears_context_window(self):
        translator = self.make_translator()
        translator._history_window = object()

        translator.updateParam("context mode", "history")

        self.assertIsNone(translator._history_window)


if __name__ == "__main__":
    unittest.main()
