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
