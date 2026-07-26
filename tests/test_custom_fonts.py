import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from ui.text_panel import FontFamilyComboBox
from utils import shared
from utils import font_loader


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class CustomFontTest(unittest.TestCase):
    def setUp(self):
        self.app = ensure_app()
        self.original_custom_fonts = shared.CUSTOM_FONTS
        self.original_custom_font_options = getattr(shared, "CUSTOM_FONT_OPTIONS", [])
        self.original_font_families = shared.FONT_FAMILIES
        shared.CUSTOM_FONTS = ["Folder Sans", "Folder Serif"]
        shared.FONT_FAMILIES = {"Folder Sans", "Folder Serif", "System Sans"}

    def tearDown(self):
        shared.CUSTOM_FONTS = self.original_custom_fonts
        shared.CUSTOM_FONT_OPTIONS = self.original_custom_font_options
        shared.FONT_FAMILIES = self.original_font_families

    def test_same_family_regular_and_bold_faces_remain_separate(self):
        option_class = getattr(font_loader, "CustomFontOption", None)
        self.assertIsNotNone(option_class)
        options = font_loader.unique_custom_font_options(
            [
                option_class("NanumGothicOTF", "Regular", False, False, "NanumGothic.otf"),
                option_class("NanumGothicOTF", "Bold", True, False, "NanumGothicBold.otf"),
            ]
        )

        self.assertEqual(
            [option.display_name for option in options],
            ["NanumGothicOTF", "NanumGothicOTF (Bold)"],
        )

    def test_selecting_bold_face_applies_family_and_bold_style(self):
        option_class = getattr(font_loader, "CustomFontOption", None)
        self.assertIsNotNone(option_class)
        shared.CUSTOM_FONTS = ["NanumGothicOTF"]
        options = [
            option_class("NanumGothicOTF", "Regular", False, False, "NanumGothic.otf"),
            option_class("NanumGothicOTF", "Bold", True, False, "NanumGothicBold.otf"),
        ]
        combo = FontFamilyComboBox()
        changes = []
        combo.param_changed.connect(lambda name, value: changes.append((name, value)))
        combo.update_font_list(options)

        combo.setCurrentIndex(1)

        self.assertEqual(
            changes,
            [("font_family", "NanumGothicOTF"), ("bold", True), ("italic", False)],
        )

    def test_font_family_names_are_deduplicated_and_sorted(self):
        self.assertTrue(hasattr(font_loader, "unique_font_families"))
        families = font_loader.unique_font_families(
            ["Pretendard Variable", "Folder Sans", "Pretendard Variable", "folder sans", ""]
        )

        self.assertEqual(families, ["Folder Sans", "Pretendard Variable"])

    def test_loaded_font_files_use_real_family_names_without_duplicates(self):
        root = Path(__file__).resolve().parents[1]
        families = font_loader.load_custom_font_families(
            [
                root / "fonts" / "PretendardVariable.ttf",
                root / "fonts" / "Hakgyoansim_ManitoOTFR.otf",
            ]
        )

        self.assertEqual(
            families,
            ["Hakgyoansim Manito OTF R", "Pretendard Variable"],
        )

    def test_invalid_family_resolves_to_a_font_from_the_fonts_folder(self):
        self.assertTrue(hasattr(font_loader, "resolve_custom_font_family"))

        self.assertEqual(
            font_loader.resolve_custom_font_family(
                "System Sans", ["Folder Serif", "Pretendard Variable"]
            ),
            "Pretendard Variable",
        )
        self.assertEqual(
            font_loader.resolve_custom_font_family(
                "folder serif", ["Folder Serif", "Pretendard Variable"]
            ),
            "Folder Serif",
        )

    def test_combo_only_lists_fonts_from_fonts_folder(self):
        combo = FontFamilyComboBox()
        combo.update_font_list(["Folder Serif", "Folder Sans", "Folder Sans"])

        self.assertEqual(
            [combo.itemText(index) for index in range(combo.count())],
            ["Folder Sans", "Folder Serif"],
        )
        self.assertIn(combo.currentText(), shared.CUSTOM_FONTS)

    def test_combo_displays_the_saved_custom_font_after_initial_load(self):
        combo = FontFamilyComboBox()

        combo.update_font_list(shared.CUSTOM_FONTS, preferred_font="Folder Serif")

        self.assertEqual(combo.currentText(), "Folder Serif")

    def test_combo_never_emits_a_system_font(self):
        combo = FontFamilyComboBox()
        combo.update_font_list(shared.CUSTOM_FONTS)
        changes = []
        combo.param_changed.connect(lambda _name, family: changes.append(family))

        combo.setCurrentText("System Sans")
        combo.apply_fontfamily()

        self.assertNotIn("System Sans", changes)
        self.assertIn(combo.currentText(), shared.CUSTOM_FONTS)


if __name__ == "__main__":
    unittest.main()
