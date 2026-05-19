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
from modules.prepare_local_files import download_and_check_hf_model_files, should_prepare_hf_model
from modules.translators import TRANSLATORS
from modules.translators import gemma4_worker
from modules.translators.trans_gemma4 import Gemma4E4BTranslator, Qwen35NineBGGUFTranslator
from modules.translators.trans_llm_api_json import (
    LLM_PROVIDER_DEFAULT_MODELS,
    LLM_PROVIDER_MODEL_OPTIONS,
    GoogleLLMTranslator,
    OpenAILLMTranslator,
)
from modules.translators.trans_nllb import NLLB200DistilledTranslator


class FakeNLLBTokenizer:
    def __init__(self):
        self.src_lang = None
        self.calls = []
        self.target_tokens = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"texts": texts}

    def convert_tokens_to_ids(self, token):
        self.target_tokens.append(token)
        return 99

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        return [f"decoded:{token}" for token in generated_tokens]


class FakeNLLBModel:
    def __init__(self):
        self.device = None
        self.generate_calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [f"gen:{text}" for text in kwargs["texts"]]


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
        prompt = kwargs["messages"][1]["content"]
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
        prompt = kwargs["messages"][1]["content"]
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
        prompt = kwargs["messages"][1]["content"]
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


class LocalTranslatorRegistrationTest(unittest.TestCase):
    def test_registers_local_translators(self):
        init_translator_registries()

        self.assertIn("NLLB-200 distilled 1.3B", TRANSLATORS.module_dict)
        self.assertIn("Gemma 4 E4B-it", TRANSLATORS.module_dict)
        self.assertIn("Qwen3.5 9B GGUF", TRANSLATORS.module_dict)


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
    def test_setup_script_selects_qwen35_from_cli_alias(self):
        with patch.object(sys, "argv", ["setup_gemma4_runtime.py", "--model", "qwen3.5"]), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(setup_gemma4_runtime.selected_model_key(), "qwen35")
            self.assertEqual(setup_gemma4_runtime.selected_quantization("qwen35"), "Q4_K_M")

    def test_setup_script_downloads_qwen35_to_declared_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = setup_gemma4_runtime.MODEL_CONFIGS["qwen35"]
            original_model_dir = config["model_dir"]
            config["model_dir"] = Path(temp_dir)
            try:
                with patch.object(sys, "argv", ["setup_gemma4_runtime.py", "--model", "qwen3.5", "--download-model"]), \
                     patch("scripts.setup_gemma4_runtime.run") as run_mock:
                    setup_gemma4_runtime.download_model(Path("/fake/python"), "qwen35")
            finally:
                config["model_dir"] = original_model_dir

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], Path("/fake/python"))
        self.assertEqual(command[1], "-c")
        self.assertIn("unsloth/Qwen3.5-9B-GGUF", command[2])
        self.assertIn("Qwen3.5-9B-Q4_K_M.gguf", command[2])


class NLLBTranslatorTest(unittest.TestCase):
    def test_translates_small_batches_with_target_language_token(self):
        tokenizer = FakeNLLBTokenizer()
        model = FakeNLLBModel()

        with patch("modules.translators.trans_nllb.osp.isdir", return_value=True), \
             patch("modules.translators.trans_nllb.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("modules.translators.trans_nllb.AutoModelForSeq2SeqLM.from_pretrained", return_value=model):
            translator = NLLB200DistilledTranslator(
                "日本語",
                "한국어",
                **{"batch size": 2},
            )
            result = translator.translate(["line one", "", "line two"])

        self.assertEqual(result, ["decoded:gen:line one", "", "decoded:gen:line two"])
        self.assertEqual(tokenizer.src_lang, "jpn_Jpan")
        self.assertEqual(tokenizer.target_tokens, ["kor_Hang"])
        self.assertEqual(model.generate_calls[0]["forced_bos_token_id"], 99)
        self.assertEqual(len(model.generate_calls), 1)


class GemmaTranslatorTest(unittest.TestCase):
    def setUp(self):
        FakeLlama.init_calls = []
        FakeLlama.completion_calls = []

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
            result = translator.translate(["line one", ""])

        self.assertEqual(result, ["one", ""])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["texts"], ["line one", ""])
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

    def test_qwen_gguf_uses_q4_model_file(self):
        stdout = '{"translations":["one"]}'
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.subprocess.run", return_value=FakeCompletedProcess(stdout=stdout)) as run_mock:
            translator = Qwen35NineBGGUFTranslator(
                "日本語",
                "한국어",
                **{
                    "worker python": "/fake/python",
                    "device": "cpu",
                    "max input tokens": 128,
                    "max new tokens": 64,
                    "context tokens": 512,
                    "gpu layers": -1,
                },
            )
            result = translator.translate(["line one"])

        self.assertEqual(result, ["one"])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["model_quantization"], "Q4_K_M")
        self.assertEqual(payload["model_log_name"], "Qwen3.5 GGUF")
        self.assertEqual(payload["model_path"], "data/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(payload["gpu_layers"], 0)

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
        payload = {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "texts": ["line one", "", "line two", "line three"],
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
        }
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
        payload = {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "texts": [f"長いセリフ{i} " + ("あ" * 240) for i in range(8)],
            "source_lang": "Japanese",
            "target_lang": "Korean",
            "max_input_tokens": 900,
            "max_new_tokens": 64,
            "context_tokens": 512,
            "gpu_layers": 0,
            "threads": 0,
            "temperature": 0.15,
            "thinking_mode": True,
            "structure_retry_count": 1,
            "chunk_context_cells": 2,
            "style_guide": "",
        }
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", FakeLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(len(result), 8)
        self.assertGreater(len(FakeLlama.completion_calls), 1)
        prompts = [call["messages"][1]["content"] for call in FakeLlama.completion_calls]
        self.assertTrue(any("Nearby page context only" in prompt for prompt in prompts))

    def test_worker_retries_strict_then_splits_failed_structure(self):
        payload = {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "texts": ["a", "b", "c", "d"],
            "source_lang": "Japanese",
            "target_lang": "Korean",
            "max_input_tokens": 4096,
            "max_new_tokens": 64,
            "context_tokens": 512,
            "gpu_layers": 0,
            "threads": 0,
            "temperature": 0.15,
            "thinking_mode": True,
            "structure_retry_count": 1,
            "chunk_context_cells": 2,
            "style_guide": "",
        }
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", RetryThenSplitLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["fixed-1", "fixed-2", "fixed-3", "fixed-4"])
        self.assertEqual(len(FakeLlama.completion_calls), 4)
        self.assertEqual(FakeLlama.completion_calls[1]["temperature"], 0.0)
        self.assertIn("strict retry", FakeLlama.completion_calls[1]["messages"][0]["content"])

    def test_worker_repairs_suspicious_single_cell_translation(self):
        payload = {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "texts": ["ありがとうありがとう", "大丈夫"],
            "source_lang": "Japanese",
            "target_lang": "Korean",
            "max_input_tokens": 4096,
            "max_new_tokens": 64,
            "context_tokens": 512,
            "gpu_layers": 0,
            "threads": 0,
            "temperature": 0.15,
            "thinking_mode": True,
            "structure_retry_count": 1,
            "chunk_context_cells": 2,
            "style_guide": "",
        }
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", SuspiciousRepairLlama):
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["고마워", "괜찮아"])
        self.assertEqual(len(FakeLlama.completion_calls), 2)
        self.assertEqual(FakeLlama.completion_calls[1]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
