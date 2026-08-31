import subprocess
import sys
import textwrap
import unittest


class LazyRuntimeTest(unittest.TestCase):
    def test_lazy_registry_keeps_torch_unimported_and_preserves_metadata(self):
        script = textwrap.dedent(
            """
            import sys
            import modules
            from modules.base import init_module_registries
            from modules.lazy_registry import validate_lazy_module_specs
            from utils.registry import ModuleSpec

            init_module_registries()
            ctd = modules.TEXTDETECTORS.get('ctd')
            mit = modules.OCR.get('mit48px')
            llm = modules.OCR.get('LLM OCR OpenAI')
            assert isinstance(ctd, ModuleSpec), type(ctd)
            assert isinstance(mit, ModuleSpec), type(mit)
            assert isinstance(llm, ModuleSpec), type(llm)
            assert 'torch' in ctd.dependencies
            assert 'torch' in mit.dependencies
            assert llm.params and 'model' in llm.params
            specs = [
                value
                for registry in modules.MODULETYPE_TO_REGISTRIES.values()
                for value in registry.module_dict.values()
                if isinstance(value, ModuleSpec)
            ]
            assert validate_lazy_module_specs(specs) == []

            for registry, key in (
                (modules.TEXTDETECTORS, 'comic_text_bubble'),
                (modules.OCR, 'one_ocr'),
                (modules.INPAINTERS, 'sdxl_inpaint'),
                (modules.TRANSLATORS, 'google'),
            ):
                spec = registry.get(key)
                assert isinstance(spec, ModuleSpec), (key, type(spec))
                assert spec.params, key

            llm_translator = modules.TRANSLATORS.get('LLM OpenAI')
            assert llm_translator.supported_src_list and llm_translator.supported_tgt_list

            expected_efforts = ['default', 'minimal', 'low', 'medium', 'high']
            llm_google = modules.TRANSLATORS.get('LLM Google')
            llm_openai = modules.TRANSLATORS.get('LLM OpenAI')
            ocr_google = modules.OCR.get('LLM OCR Google')
            ocr_openai = modules.OCR.get('LLM OCR OpenAI')
            assert llm_google.params['reasoning effort']['options'] == expected_efforts
            assert ocr_google.params['reasoning effort']['options'] == expected_efforts
            assert 'reasoning effort' not in llm_openai.params
            assert 'reasoning effort' not in ocr_openai.params
            gemini_36 = 'GGL: gemini-3.6-flash'
            assert gemini_36 in llm_google.params['model']['options']
            assert gemini_36 in ocr_google.params['model']['options']

            comic = modules.TEXTDETECTORS.get('comic_text_bubble')
            assert comic.hf_model_repo_id == 'ogkalu/comic-text-and-bubble-detector'
            assert comic.hf_model_download_on_prepare is True
            assert 'torch' not in sys.modules
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
