import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

from modules.base import BaseModule, init_translator_registries
from modules.prepare_local_files import download_and_check_hf_model_files, should_prepare_hf_model
from modules.translators import TRANSLATORS
from modules.translators import gemma4_worker
from modules.translators.trans_gemma4 import Gemma4E4BTranslator
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
        return {
            "choices": [
                {
                    "message": {
                        "content": f"translation-{len(self.completion_calls)}",
                    }
                }
            ]
        }


class LocalTranslatorRegistrationTest(unittest.TestCase):
    def test_registers_local_translators(self):
        init_translator_registries()

        self.assertIn("NLLB-200 distilled 1.3B", TRANSLATORS.module_dict)
        self.assertIn("Gemma 4 E4B-it", TRANSLATORS.module_dict)


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
                },
            )
            result = translator.translate(["line one", ""])

        self.assertEqual(result, ["one", ""])
        payload = json.loads(run_mock.call_args.kwargs["input"])
        self.assertEqual(payload["texts"], ["line one", ""])
        self.assertEqual(payload["source_lang"], "Japanese")
        self.assertEqual(payload["target_lang"], "Korean")
        self.assertEqual(payload["model_path"], "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf")
        self.assertEqual(payload["gpu_layers"], 0)
        self.assertEqual(payload["context_tokens"], 512)

    def test_missing_gguf_model_returns_short_error(self):
        with patch("modules.translators.trans_gemma4.osp.isfile", return_value=False), \
             patch("modules.translators.trans_gemma4.subprocess.run") as run_mock:
            translator = Gemma4E4BTranslator("日本語", "한국어", **{"worker python": "/fake/python"})
            result = translator.translate(["line one", ""])

        self.assertEqual(result[1], "")
        self.assertIn("Gemma4 GGUF Q4_K_M model file is missing", result[0])
        run_mock.assert_not_called()

    def test_missing_worker_returns_short_error(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("modules.translators.trans_gemma4.osp.isfile", return_value=True), \
             patch("modules.translators.trans_gemma4.Path.exists", return_value=False):
            translator = Gemma4E4BTranslator("日本語", "한국어")
            result = translator.translate(["line one", ""])

        self.assertEqual(result[1], "")
        self.assertIn("Gemma4 GGUF runtime is not configured", result[0])

    def test_worker_translates_each_non_empty_cell_in_order_with_rolling_context(self):
        payload = {
            "model_path": "data/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
            "texts": ["line one", "", "line two", "line three"],
            "source_lang": "Japanese",
            "target_lang": "Korean",
            "max_input_tokens": 128,
            "max_new_tokens": 64,
            "context_tokens": 512,
            "gpu_layers": 0,
            "threads": 0,
            "temperature": 0.0,
            "thinking_mode": True,
        }
        with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
             patch("modules.translators.gemma4_worker.Llama", FakeLlama), \
             patch("modules.translators.gemma4_worker.gc.collect") as collect_mock:
            result = gemma4_worker.translate(payload)

        self.assertEqual(result, ["translation-1", "", "translation-2", "translation-3"])
        self.assertEqual(len(FakeLlama.completion_calls), 3)
        first_prompt = FakeLlama.completion_calls[0]["messages"][1]["content"]
        second_prompt = FakeLlama.completion_calls[1]["messages"][1]["content"]
        third_prompt = FakeLlama.completion_calls[2]["messages"][1]["content"]

        self.assertIn("Previous source text:\n(none)", first_prompt)
        self.assertIn("Previous translation:\n(none)", first_prompt)
        self.assertIn("Previous source text:\nline one", second_prompt)
        self.assertIn("Previous translation:\ntranslation-1", second_prompt)
        self.assertIn("Current source text:\nline two", second_prompt)
        self.assertIn("Previous source text:\nline two", third_prompt)
        self.assertIn("Previous translation:\ntranslation-2", third_prompt)
        self.assertNotIn("line one", third_prompt)
        self.assertNotIn("translation-1", third_prompt)
        self.assertGreaterEqual(collect_mock.call_count, 4)

    def test_worker_cleans_thinking_and_labels_from_output(self):
        cleaned = gemma4_worker._clean_translation("<think>notes</think>\nTranslation: 안녕")
        self.assertEqual(cleaned, "안녕")


if __name__ == "__main__":
    unittest.main()
