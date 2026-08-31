import os
import sys
import tempfile
import unittest
from unittest.mock import patch

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from modules.base import BaseModule, init_translator_registries
from modules.prepare_local_files import (
    download_and_check_hf_model_files,
    ensure_module_files,
    should_prepare_hf_model,
)
from modules.translators import TRANSLATORS
from modules.translators.trans_llm_api_json import (
    LLM_PROVIDER_DEFAULT_MODELS,
    LLM_PROVIDER_MODEL_OPTIONS,
    GoogleLLMTranslator,
    OpenAILLMTranslator,
)
from utils.registry import ModuleSpec


class LocalTranslatorRegistrationTest(unittest.TestCase):
    def test_retired_local_translators_are_not_registered(self):
        init_translator_registries()

        self.assertNotIn("Gemma 4 E4B-it", TRANSLATORS.module_dict)
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


if __name__ == "__main__":
    unittest.main()
