import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from scripts import setup_gemma4_runtime
from modules.base import BaseModule, init_translator_registries
from modules.prepare_local_files import (
    download_and_check_hf_model_files,
    ensure_module_files,
    should_prepare_hf_model,
)
from modules.context.glossary import GlossaryEntry
from modules.context.history import (
    HistoryPage,
    HistoryWindowKey,
    RequestContext,
)
from modules.translators import TRANSLATORS
from modules.translators import gemma4_worker
from modules.translators.trans_gemma4 import Gemma4E4BTranslator
from modules.translators.trans_llm_api_json import (
    LLM_PROVIDER_DEFAULT_MODELS,
    LLM_PROVIDER_MODEL_OPTIONS,
    GoogleLLMTranslator,
    OpenAILLMTranslator,
)
from utils.registry import ModuleSpec


class FakeInputIds:
    shape = (1, 4)

    def to(self, device):
        return self


class FakeGemmaInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeGenerated:
    def __init__(self, response):
        self.response = response

    def __getitem__(self, item):
        return self


class FakeGemmaProcessor:
    def __init__(self):
        self.chat_template_calls = []
        self.input_prompts = []
        self.tokenizer = type("FakeTokenizer", (), {"eos_token_id": 1})()

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append({"messages": messages, "kwargs": kwargs})
        return messages[-1]["content"]

    def __call__(self, *args, **kwargs):
        prompt = kwargs.get("text")
        if prompt is None and args:
            prompt = args[0]
        self.input_prompts.append(prompt)
        return FakeGemmaInputs(input_ids=FakeInputIds(), prompt=prompt)

    def decode(self, generated_tokens, skip_special_tokens=False):
        return generated_tokens.response

    def parse_response(self, response):
        return response


class FakeGenerationConfig:
    pad_token_id = None


class FakeGemmaModel:
    def __init__(self):
        self.device = None
        self.generate_calls = []
        self.generation_config = FakeGenerationConfig()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return FakeGenerated(f"translation-{len(self.generate_calls)}")


class FakeCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeLlama:
    init_calls = []
    completion_calls = []

    def __init__(self, **kwargs):
        self.init_calls.append(kwargs)

    def create_chat_completion(self, **kwargs):
        self.completion_calls.append(kwargs)
        prompt = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "user"
        )
        page_json = prompt.split("Page source texts:\n", 1)[1]
        items = json.loads(page_json)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {"id": item["id"], "translation": f"translation-{seq}"}
                                    for seq, item in enumerate(items, 1)
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    }
                }
            ]
        }


class RetryThenSplitLlama(FakeLlama):
    def create_chat_completion(self, **kwargs):
        self.completion_calls.append(kwargs)
        if len(self.completion_calls) <= 2:
            return {"choices": [{"message": {"content": "not json"}}]}
        prompt = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "user"
        )
        items = json.loads(prompt.split("Page source texts:\n", 1)[1])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {"id": item["id"], "translation": f"fixed-{item['id']}"}
                                    for item in items
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class SuspiciousRepairLlama(FakeLlama):
    def create_chat_completion(self, **kwargs):
        self.completion_calls.append(kwargs)
        prompt = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "user"
        )
        items = json.loads(prompt.split("Page source texts:\n", 1)[1])
        if len(self.completion_calls) == 1:
            translations = [
                {"id": items[0]["id"], "translation": items[0]["text"]},
                {"id": items[1]["id"], "translation": "괜찮아"},
            ]
        else:
            translations = [{"id": items[0]["id"], "translation": "고마워"}]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"translations": translations}, ensure_ascii=False)
                    }
                }
            ]
        }


class HistoryBudgetLlama(FakeLlama):
    def tokenize(self, serialized, add_bos=False):
        text = serialized.decode("utf-8")
        if "old target" in text:
            token_count = 55
        elif "recent target" in text:
            token_count = 10
        else:
            token_count = 60
        return list(range(token_count))


class NonAdditiveHistoryTokenLlama:
    def tokenize(self, serialized, add_bos=False):
        messages = json.loads(serialized.decode("utf-8"))
        token_count = 11 if len(messages) > 2 else 4
        return list(range(token_count))


class FlatHistoryTokenLlama:
    def tokenize(self, serialized, add_bos=False):
        return [0]


class LocalTranslatorRegistrationTest(unittest.TestCase):
    def test_registers_local_translators(self):
        init_translator_registries()

        self.assertIn("Gemma 4 E4B-it", TRANSLATORS.module_dict)
        self.assertNotIn("NLLB-200 distilled 1.3B", TRANSLATORS.module_dict)
        self.assertNotIn("Qwen3.5 9B GGUF", TRANSLATORS.module_dict)


class LLMTranslatorCatalogTest(unittest.TestCase):
    def test_fixed_provider_translators_use_central_model_catalog(self):
        self.assertEqual(
            OpenAILLMTranslator.params["model"]["options"],
            LLM_PROVIDER_MODEL_OPTIONS["OpenAI"],
        )
        self.assertEqual(
            OpenAILLMTranslator.params["model"]["value"],
            LLM_PROVIDER_DEFAULT_MODELS["OpenAI"],
        )
        self.assertEqual(
            GoogleLLMTranslator.params["model"]["options"],
            LLM_PROVIDER_MODEL_OPTIONS["Google"],
        )

    def test_fixed_provider_translator_uses_env_api_key_when_param_is_empty(self):
        with patch.dict(os.environ, {"BALLOONTRANS_LLM_OPENAI_API_KEY": "openai-project-key"}, clear=True):
            translator = OpenAILLMTranslator(
                "日本語",
                "한국어",
                raise_unsupported_lang=False,
                **{"apikey": ""},
            )
            self.assertEqual(translator.apikey, "openai-project-key")


class BaseModuleLoadingTest(unittest.TestCase):
    def test_model_loading_lock_is_released_when_load_fails(self):
        class FailingModule(BaseModule):
            def _load_model(self):
                raise RuntimeError("load failed")

        with patch("modules.base.aquire_model_loading_lock") as acquire_mock, \
             patch("modules.base.release_model_loading_lock") as release_mock:
            with self.assertRaises(RuntimeError):
                FailingModule().load_model()

        acquire_mock.assert_called_once()
        release_mock.assert_called_once()


class LocalModelDownloadTest(unittest.TestCase):
    def test_lazy_module_prepare_downloads_opted_in_hf_snapshot(self):
        spec = ModuleSpec(
            key="example",
            import_path="example.module",
            class_name="ExampleModule",
            hf_model_repo_id="example/model",
            hf_model_save_dir="data/models/example",
            hf_model_required_files=["config.json"],
            hf_model_download_on_prepare=True,
        )

        with patch(
            "modules.prepare_local_files.download_and_check_hf_model_files",
            return_value=True,
        ) as download_mock:
            self.assertTrue(ensure_module_files(spec))

        download_mock.assert_called_once_with(
            spec,
            progress_callback=None,
            cancel_event=None,
        )

    def test_downloads_missing_hf_snapshot_to_declared_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            class FakeModule:
                hf_model_repo_id = "example/model"
                hf_model_save_dir = temp_dir
                hf_model_required_files = ["config.json", ["*.safetensors", "pytorch_model.bin"]]
                hf_model_ignore_patterns = ["*.h5"]

            def fake_snapshot_download(**kwargs):
                with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf8") as f:
                    f.write("{}")
                with open(os.path.join(temp_dir, "model.safetensors"), "w", encoding="utf8") as f:
                    f.write("fake")
                return temp_dir

            with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download) as download_mock:
                self.assertTrue(download_and_check_hf_model_files(FakeModule))

            download_mock.assert_called_once()
            self.assertEqual(download_mock.call_args.kwargs["repo_id"], "example/model")
            self.assertEqual(download_mock.call_args.kwargs["local_dir"], temp_dir)
            self.assertEqual(download_mock.call_args.kwargs["ignore_patterns"], ["*.h5"])

    def test_skips_hf_download_when_required_snapshot_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf8") as f:
                f.write("{}")
            with open(os.path.join(temp_dir, "model.safetensors"), "w", encoding="utf8") as f:
                f.write("fake")

            class FakeModule:
                hf_model_repo_id = "example/model"
                hf_model_save_dir = temp_dir
                hf_model_required_files = ["config.json", ["*.safetensors", "pytorch_model.bin"]]

            with patch("huggingface_hub.snapshot_download") as download_mock:
                self.assertTrue(download_and_check_hf_model_files(FakeModule))

            download_mock.assert_not_called()

    def test_hf_snapshots_are_not_prepared_by_default(self):
        class FakeModule:
            hf_model_repo_id = "example/model"
            hf_model_save_dir = "data/models/example"

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(should_prepare_hf_model(FakeModule))

    def test_hf_snapshot_prepare_can_be_enabled_by_env(self):
        class FakeModule:
            hf_model_repo_id = "example/model"
            hf_model_save_dir = "data/models/example"

        with patch.dict(os.environ, {"BALLOONTRANS_DOWNLOAD_HF_MODEL_ON_PREPARE": "1"}):
            self.assertTrue(should_prepare_hf_model(FakeModule))


class GGUFSetupRuntimeTest(unittest.TestCase):
    def test_setup_script_only_exposes_gemma4_model_config(self):
        self.assertEqual(set(setup_gemma4_runtime.MODEL_CONFIGS), {"gemma4"})

    def test_setup_script_selects_gemma4_from_cli_alias(self):
        with patch.object(sys, "argv", ["setup_gemma4_runtime.py", "--model", "gemma4"]), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(setup_gemma4_runtime.selected_model_key(), "gemma4")
            self.assertEqual(setup_gemma4_runtime.selected_quantization("gemma4"), "Q4_K_M")

    def test_setup_script_downloads_gemma4_to_declared_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = setup_gemma4_runtime.MODEL_CONFIGS["gemma4"]
            original_model_dir = config["model_dir"]
            config["model_dir"] = Path(temp_dir)
            try:
                with patch.object(sys, "argv", ["setup_gemma4_runtime.py", "--model", "gemma4", "--download-model"]), \
                     patch("scripts.setup_gemma4_runtime.run") as run_mock:
                    setup_gemma4_runtime.download_model(Path("/fake/python"), "gemma4")
            finally:
                config["model_dir"] = original_model_dir

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], Path("/fake/python"))
        self.assertEqual(command[1], "-c")
        self.assertIn("unsloth/gemma-4-E4B-it-GGUF", command[2])
        self.assertIn("gemma-4-E4B-it-Q4_K_M.gguf", command[2])


class GemmaTranslatorTest(unittest.TestCase):
    def setUp(self):
        FakeLlama.init_calls = []
        FakeLlama.completion_calls = []

    @staticmethod
    def base_worker_payload():
        return {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "model_quantization": "Q4_K_M",
            "model_log_name": "Gemma4 GGUF",
            "texts": ["line one"],
            "source_lang": "Japanese",
            "target_lang": "Korean",
            "max_input_tokens": 4096,
            "max_new_tokens": 64,
            "context_tokens": 512,
            "gpu_layers": 0,
            "threads": 0,
            "temperature": 0.15,
            "top_p": 0.8,
            "top_k": 32,
            "thinking_mode": True,
            "structure_retry_count": 1,
            "chunk_context_cells": 2,
            "style_guide": "",
            "history_pages": [],
            "history_token_budget": 0,
            "glossary_json": "",
            "glossary_mode": "matching",
        }

    def test_subprocess_runtime_calls_worker_with_gguf_payload(self):
        stdout = '{"translations":["one",""]}'
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.subprocess.run", return_value=FakeCompletedProcess(stdout=stdout)) as run_mock:
            translator = Gemma4E4BTranslator(
                "日本語",
                "한국어",
                **{
                    "worker python": "/fake/python",
                    "device": "cpu",
                    "max input tokens": 128,
                    "max new tokens": 64,
                    "context tokens": 512,
                    "gpu layers": -1,
                    "top_p": 0.8,
                    "top_k": 32,
                },
            )
            history_page = translator._render_history_page(
                HistoryPage("001.png", ("Hero",), ("용사",))
            )
            request_context = RequestContext(
                history=(history_page,),
                glossary=(GlossaryEntry("Hero", "용사"),),
                glossary_mode="matching",
                history_budget=4096,
                window_key=HistoryWindowKey(
                    object(),
                    (("model", "gemma"),),
                ),
                request_page_key="002.png",
            )
            result = translator._translate(
                ["Hero", ""],
                request_context=request_context,
            )

        self.assertEqual(result, ["one", ""])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["texts"], ["Hero", ""])
        self.assertEqual(payload["source_lang"], "Japanese")
        self.assertEqual(payload["target_lang"], "Korean")
        self.assertEqual(payload["model_path"], "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")
        self.assertEqual(payload["model_quantization"], "Q4_K_M")
        self.assertEqual(payload["gpu_layers"], 0)
        self.assertEqual(payload["context_tokens"], 512)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["top_k"], 32)
        self.assertEqual(payload["structure_retry_count"], 1)
        self.assertEqual(payload["chunk_context_cells"], 2)
        self.assertIn("자연스러운 한국어", payload["style_guide"])
        self.assertEqual(payload["history_token_budget"], 4096)
        self.assertEqual(payload["history_pages"][0]["page_key"], "001.png")
        self.assertEqual(
            [
                message["role"]
                for message in payload["history_pages"][0]["messages"]
            ],
            ["user", "assistant"],
        )
        self.assertIn('"source":"Hero"', payload["glossary_json"])
        self.assertEqual(payload["glossary_mode"], "matching")
        self.assertIsNone(translator._history_window)

    def test_gemma_context_params_are_fresh_and_independent_from_openai(self):
        context_keys = (
            "context mode",
            "history token budget",
            "glossary path",
            "glossary mode",
        )

        for key in context_keys:
            self.assertIn(key, Gemma4E4BTranslator.params)
            self.assertIsNot(
                Gemma4E4BTranslator.params[key],
                OpenAILLMTranslator.params[key],
            )
        first = Gemma4E4BTranslator("日本語", "한국어")
        second = Gemma4E4BTranslator("日本語", "한국어")
        for key in context_keys:
            self.assertIsNot(first.params[key], second.params[key])

    def test_legacy_top_sampling_names_are_migrated(self):
        stdout = '{"translations":["one"]}'
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.subprocess.run", return_value=FakeCompletedProcess(stdout=stdout)) as run_mock:
            translator = Gemma4E4BTranslator(
                "日本語",
                "한국어",
                **{
                    "worker python": "/fake/python",
                    "device": "cpu",
                    "top p": 0.75,
                    "top k": 24,
                },
            )
            result = translator.translate(["line one"])

        self.assertEqual(result, ["one"])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["top_p"], 0.75)
        self.assertEqual(payload["top_k"], 24)

    def test_q6_quantization_uses_upstream_q6_file(self):
        stdout = '{"translations":["one"]}'
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.subprocess.run", return_value=FakeCompletedProcess(stdout=stdout)) as run_mock:
            translator = Gemma4E4BTranslator(
                "日本語",
                "한국어",
                **{
                    "worker python": "/fake/python",
                    "model quantization": "Q6_K_M",
                },
            )
            result = translator.translate(["line one"])

        self.assertEqual(result, ["one"])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["model_quantization"], "Q6_K_M")
        self.assertEqual(payload["model_path"], "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q6_K.gguf")

    def test_missing_gguf_model_returns_short_error(self):
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=False), \
             patch("modules.translators.trans_gemma4.subprocess.run") as run_mock:
            translator = Gemma4E4BTranslator(
                "日本語",
                "한국어",
                **{"worker python": "/fake/python", "model quantization": "Q6_K_M"},
            )
            result = translator.translate(["line one", ""])

        self.assertEqual(result[1], "")
        self.assertIn("Gemma4 GGUF Q6_K_M model file is missing", result[0])
        self.assertIn("gemma-4-E4B-it-Q6_K.gguf", result[0])
        run_mock.assert_not_called()

    def test_missing_worker_returns_short_error(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.Path.exists", return_value=False):
            translator = Gemma4E4BTranslator("日本語", "한국어")
            result = translator.translate(["line one", ""])

        self.assertEqual(result[1], "")
        self.assertIn("Gemma4 GGUF runtime is not configured", result[0])

    def test_worker_translates_full_page_in_one_generation(self):
        payload = self.base_worker_payload()
        payload["texts"] = ["line one", "", "line two", "line three"]
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", FakeLlama), \
             patch("modules.translators.gemma4_worker.gc.collect") as collect_mock:
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["translation-1", "", "translation-2", "translation-3"])
        self.assertEqual(len(FakeLlama.completion_calls), 1)
        self.assertEqual(FakeLlama.completion_calls[0]["top_p"], 0.8)
        self.assertEqual(FakeLlama.completion_calls[0]["top_k"], 32)
        page_prompt = FakeLlama.completion_calls[0]["messages"][1]["content"]
        system_prompt = FakeLlama.completion_calls[0]["messages"][0]["content"]

        self.assertIn("Treat all cells as shared page context", page_prompt)
        self.assertIn('"id": 1', page_prompt)
        self.assertIn('"text": "line one"', page_prompt)
        self.assertIn('"id": 3', page_prompt)
        self.assertIn('"text": "line two"', page_prompt)
        self.assertIn('"id": 4', page_prompt)
        self.assertIn('"text": "line three"', page_prompt)
        self.assertNotIn("Previous source text", page_prompt)
        self.assertIn("highest priority is natural, fluent dialogue", system_prompt)
        self.assertIn("character voice", system_prompt)
        self.assertIn("자연스러운 한국어", page_prompt)
        self.assertIn("말투/존댓말", page_prompt)
        self.assertIn("말풍선 길이", page_prompt)
        self.assertIn("OCR 잡음 보정", page_prompt)
        self.assertIn("JSON object", page_prompt)
        self.assertGreaterEqual(collect_mock.call_count, 2)

    def test_worker_cleans_thinking_and_labels_from_output(self):
        cleaned = gemma4_worker._clean_translation("<think>notes</think>\nTranslation: 안녕")
        self.assertEqual(cleaned, "안녕")

    def test_worker_splits_large_pages_by_max_input_tokens(self):
        payload = self.base_worker_payload()
        payload.update({
            "texts": [f"長いセリフ{i} " + ("あ" * 240) for i in range(8)],
            "max_input_tokens": 900,
        })
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", FakeLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(len(result), 8)
        self.assertGreater(len(FakeLlama.completion_calls), 1)
        prompts = [call["messages"][1]["content"] for call in FakeLlama.completion_calls]
        self.assertTrue(any("Nearby page context only" in prompt for prompt in prompts))

    def test_worker_retries_strict_then_splits_failed_structure(self):
        payload = self.base_worker_payload()
        payload.update({
            "texts": ["a", "b", "c", "d"],
            "history_pages": [{
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "recent target"},
                ],
            }],
            "history_token_budget": 4096,
            "glossary_json": '{"glossary":[{"source":"a","translation":"에이","note":""}]}',
            "glossary_mode": "matching",
        })
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", RetryThenSplitLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["fixed-1", "fixed-2", "fixed-3", "fixed-4"])
        self.assertEqual(len(FakeLlama.completion_calls), 4)
        self.assertEqual(FakeLlama.completion_calls[1]["temperature"], 0.0)
        self.assertIn("strict retry", FakeLlama.completion_calls[1]["messages"][0]["content"])
        for call in FakeLlama.completion_calls:
            serialized = "\n".join(
                message["content"] for message in call["messages"]
            )
            self.assertIn("recent target", serialized)
            self.assertIn('"source":"a"', call["messages"][-1]["content"])

    def test_worker_repairs_suspicious_single_cell_translation(self):
        payload = self.base_worker_payload()
        payload.update({
            "texts": ["ありがとうありがとう", "大丈夫"],
            "history_pages": [{
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "recent target"},
                ],
            }],
            "history_token_budget": 4096,
            "glossary_json": '{"glossary":[{"source":"ありがとう","translation":"고마워","note":""}]}',
            "glossary_mode": "matching",
        })
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", SuspiciousRepairLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["고마워", "괜찮아"])
        self.assertEqual(len(FakeLlama.completion_calls), 2)
        self.assertEqual(FakeLlama.completion_calls[1]["temperature"], 0.0)
        for call in FakeLlama.completion_calls:
            serialized = "\n".join(
                message["content"] for message in call["messages"]
            )
            self.assertIn("recent target", serialized)
            self.assertIn(
                '"source":"ありがとう"',
                call["messages"][-1]["content"],
            )

    def test_worker_places_full_glossary_before_history_and_current_page(self):
        payload = self.base_worker_payload()
        payload.update({
            "history_pages": [{
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "previous target"},
                ],
            }],
            "history_token_budget": 4096,
            "glossary_json": '{"glossary":[{"source":"Hero","translation":"용사","note":""}]}',
            "glossary_mode": "all",
        })

        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", FakeLlama):
            gemma4_worker.translate(payload)

        messages = FakeLlama.completion_calls[0]["messages"]
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user", "assistant", "user"],
        )
        self.assertIn("glossary", messages[1]["content"])
        self.assertNotIn("glossary", messages[-1]["content"])

    def test_worker_drops_oldest_whole_history_pages_before_current_chunk(self):
        payload = self.base_worker_payload()
        payload.update({
            "max_input_tokens": 120,
            "history_token_budget": 40,
            "history_pages": [
                {
                    "page_key": "001.png",
                    "messages": [
                        {"role": "user", "content": "old " * 30},
                        {"role": "assistant", "content": "old target " * 20},
                    ],
                },
                {
                    "page_key": "002.png",
                    "messages": [
                        {"role": "user", "content": "recent"},
                        {"role": "assistant", "content": "recent target"},
                    ],
                },
            ],
        })

        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", HistoryBudgetLlama):
            gemma4_worker.translate(payload)

        messages = FakeLlama.completion_calls[0]["messages"]
        serialized = "\n".join(message["content"] for message in messages)
        self.assertNotIn("old target", serialized)
        self.assertIn("recent target", serialized)
        self.assertIn("Page source texts", serialized)

    def test_worker_keeps_current_page_and_glossary_when_history_cannot_fit(self):
        payload = self.base_worker_payload()
        payload.update({
            "max_input_tokens": 59,
            "history_token_budget": 4096,
            "history_pages": [{
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "recent target"},
                ],
            }],
            "glossary_json": '{"glossary":[{"source":"line","translation":"줄","note":""}]}',
            "glossary_mode": "matching",
        })

        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", HistoryBudgetLlama):
            gemma4_worker.translate(payload)

        messages = FakeLlama.completion_calls[0]["messages"]
        serialized = "\n".join(message["content"] for message in messages)
        self.assertNotIn("recent target", serialized)
        self.assertIn('"source":"line"', messages[-1]["content"])
        self.assertIn('"text": "line one"', messages[-1]["content"])

    def test_history_fitting_uses_exact_combined_message_token_count(self):
        base_messages = [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "current page"},
        ]
        payload = {
            "max_input_tokens": 10,
            "history_token_budget": 10,
            "history_pages": [{
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "previous target"},
                ],
            }],
        }

        fitted = gemma4_worker._fit_history_pages_to_budget(
            NonAdditiveHistoryTokenLlama(),
            payload,
            base_messages,
        )

        self.assertEqual(fitted, [])

    def test_history_fitting_rejects_malformed_pages_as_whole_pairs(self):
        valid_messages = [
            {"role": "user", "content": "valid source"},
            {"role": "assistant", "content": "valid target"},
        ]
        payload = {
            "max_input_tokens": 100,
            "history_token_budget": 100,
            "history_pages": [
                {
                    "messages": [
                        {"role": "user", "content": "missing key source"},
                        {"role": "assistant", "content": "missing key target"},
                    ],
                },
                {
                    "page_key": "",
                    "messages": [
                        {"role": "user", "content": "empty key source"},
                        {"role": "assistant", "content": "empty key target"},
                    ],
                },
                {
                    "page_key": "wrong-cardinality.png",
                    "messages": [
                        {"role": "user", "content": "only one message"},
                    ],
                },
                {
                    "page_key": "wrong-roles.png",
                    "messages": [
                        {"role": "system", "content": "system source"},
                        {"role": "assistant", "content": "assistant target"},
                    ],
                },
                {
                    "page_key": "reversed-roles.png",
                    "messages": [
                        {"role": "assistant", "content": "reversed target"},
                        {"role": "user", "content": "reversed source"},
                    ],
                },
                {
                    "page_key": "empty-content.png",
                    "messages": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": "empty target"},
                    ],
                },
                {
                    "page_key": "empty-assistant.png",
                    "messages": [
                        {"role": "user", "content": "empty assistant source"},
                        {"role": "assistant", "content": ""},
                    ],
                },
                {
                    "page_key": "whitespace-assistant.png",
                    "messages": [
                        {"role": "user", "content": "whitespace assistant source"},
                        {"role": "assistant", "content": " \t "},
                    ],
                },
                {
                    "page_key": "valid.png",
                    "messages": valid_messages,
                },
            ],
        }
        base_messages = [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "current page"},
        ]

        fitted = gemma4_worker._fit_history_pages_to_budget(
            FlatHistoryTokenLlama(),
            payload,
            base_messages,
        )

        self.assertEqual(fitted, valid_messages)


if __name__ == "__main__":
    unittest.main()
