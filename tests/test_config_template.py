import json
import os
import tempfile
import unittest
from pathlib import Path

from utils.config import ProgramConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = PROJECT_ROOT / "config" / "config.json"


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


class ConfigTemplateTest(unittest.TestCase):
    def load_template_dict(self):
        with CONFIG_TEMPLATE.open("r", encoding="utf8") as config_file:
            return json.load(config_file)

    def test_config_template_loads_as_program_config(self):
        config = ProgramConfig.load(str(CONFIG_TEMPLATE))

        self.assertEqual(config.module.translator, "google")
        self.assertEqual(config.module.ocr, "mit48px")
        self.assertEqual(config.recent_proj_list, [])
        self.assertFalse(config.package_manager.auto_install_missing_packages)

    def test_config_template_does_not_persist_local_state(self):
        config = self.load_template_dict()

        self.assertEqual(config["recent_proj_list"], [])
        self.assertFalse(os.path.isabs(config["text_styles_path"]))
        self.assertEqual(config["text_styles_path"], "config/textstyles/default.json")

        module_config = config["module"]
        self.assertEqual(module_config["textdetector_params"], {})
        self.assertEqual(module_config["ocr_params"], {})
        self.assertEqual(module_config["translator_params"], {})
        self.assertEqual(module_config["inpainter_params"], {})

        local_markers = [
            str(PROJECT_ROOT),
            "/Volumes/",
            "/Users/",
            "\\Users\\",
            "Hakgyoansim Manito",
            "\"mps\"",
        ]
        serialized = json.dumps(config, ensure_ascii=False)
        for marker in local_markers:
            self.assertNotIn(marker, serialized)

    def test_retired_gemma_config_migrates_to_google_and_drops_stale_params(self):
        legacy_config = self.load_template_dict()
        legacy_config["module"]["translator"] = "Gemma 4 E4B-it"
        legacy_config["module"]["translator_params"] = {
            "Gemma 4 E4B-it": {"model quantization": "Q6_K_M"},
            "Papago": {"delay": 1.0},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "legacy-config.json"
            config_path.write_text(
                json.dumps(legacy_config, ensure_ascii=False),
                encoding="utf8",
            )
            config = ProgramConfig.load(str(config_path))

        self.assertEqual(config.module.translator, "google")
        self.assertNotIn(
            "Gemma 4 E4B-it",
            config.module.translator_params,
        )
        self.assertEqual(
            config.module.translator_params["Papago"],
            {"delay": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
