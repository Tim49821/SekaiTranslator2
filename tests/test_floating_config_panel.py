import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt, QPoint, QEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QStackedWidget, QWidget, QLabel, QCheckBox

from ui.configpanel import ConfigPanel
from ui.mainwindow import MainWindow
from ui.mainwindowbars import StateChecker
from ui.drawingpanel import InpaintPanel, RectPanel
from ui.module_parse_widgets import ParamLineEditor, ParamEditor, ParamCheckGroup
from utils.config import pcfg


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FloatingConfigPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def tearDown(self):
        for widget in QApplication.topLevelWidgets():
            widget.close()
        self.app.processEvents()

    def detector_panel(self):
        panel = ConfigPanel()
        detector = panel.detect_config_panel
        detector.get_valid_module_keys = lambda: ['ctd']
        detector.addModulesParamWidgets({'ctd': {
            'detect_size': {'type': 'selector', 'options': [1024, 1280], 'value': 1280},
            'det_rearrange_max_batches': {'type': 'selector', 'options': [4, 32], 'value': 4},
            'device': {'type': 'selector', 'options': ['cpu', 'mps'], 'value': 'mps'},
            'font size multiplier': 1.0, 'font size max': -1,
            'font size min': -1, 'mask dilate size': 2,
        }})
        panel.show()
        panel.focusOnDetect()
        self.app.processEvents()
        self.app.processEvents()
        return panel

    def test_navigation_shows_only_selected_section_at_top(self):
        panel = self.detector_panel()
        panel.configTable.setCurrentItem(0, 1)
        self.app.processEvents()
        self.assertTrue(panel.ocr_config_panel.isVisible())
        self.assertFalse(panel.detect_config_panel.isVisible())
        self.assertFalse(panel.load_model_checker.isVisible())
        self.assertEqual(panel.configContent.verticalScrollBar().value(), 0)

    def test_detector_fields_fit_without_scrolling_and_align_with_model(self):
        panel = self.detector_panel()
        content = panel.configContent
        self.assertEqual(content.horizontalScrollBar().maximum(), 0)
        self.assertEqual(content.verticalScrollBar().maximum(), 0)
        model = panel.detect_config_panel.module_combobox
        expected_right = model.mapTo(panel, model.rect().topRight()).x()
        for editor in panel.detect_config_panel.findChildren(ParamLineEditor):
            self.assertEqual(editor.mapTo(panel, editor.rect().topRight()).x(), expected_right)

    def test_each_settings_page_fits_without_horizontal_scrolling(self):
        panel = self.detector_panel()
        for width in (900, 800):
            panel.resize(width, 700)
            for group in (0, 1):
                for section in (-1, 0, 1, 2, 3):
                    with self.subTest(width=width, group=group, section=section):
                        panel.configTable.setCurrentItem(group, section)
                        self.app.processEvents()
                        self.app.processEvents()
                        self.assertEqual(panel.configContent.horizontalScrollBar().maximum(), 0)

    def test_module_hover_does_not_tint_content_or_change_navigation(self):
        panel = self.detector_panel()
        block = panel.detect_sub_block
        QApplication.sendEvent(block, QEvent(QEvent.Type.Enter))
        self.app.processEvents()
        # A point in the empty row margin, away from labels and controls.
        color = block.grab().toImage().pixelColor(2, 2)
        self.assertGreaterEqual(min(color.red(), color.green(), color.blue()), 248)
        self.assertEqual(panel.configTable.currentIndex().row(), 0)

    def test_long_editor_fits_and_edits_survive_page_switch(self):
        panel = ConfigPanel()
        ocr = panel.ocr_config_panel
        ocr.get_valid_module_keys = lambda: ['example']
        ocr.addModulesParamWidgets({'example': {
            'Long option name that must wrap instead of clipping the input': {
                'type': 'selector', 'options': ['A long module option ' * 6],
                'value': 'A long module option ' * 6, 'size': 'long'},
            'prompt': {'type': 'editor', 'value': 'Initial prompt'},
        }})
        panel.show()
        panel.focusOnOCR()
        self.app.processEvents()
        editor = ocr.findChild(ParamEditor)
        editor.setText('Edited prompt')
        panel.focusOnDetect()
        panel.focusOnOCR()
        self.app.processEvents()
        self.assertEqual(editor.text(), 'Edited prompt')
        self.assertEqual(panel.configContent.horizontalScrollBar().maximum(), 0)

    def test_page_title_stays_visible_when_long_settings_are_scrolled(self):
        panel = self.detector_panel()
        panel.resize(900, 500)
        self.app.processEvents()
        title = next(label for label in panel.findChildren(QLabel)
                     if label.isVisible() and label.text() == 'Text Detection')
        original_y = title.mapTo(panel, QPoint()).y()
        scrollbar = panel.configContent.verticalScrollBar()
        self.assertGreater(scrollbar.maximum(), 0)
        scrollbar.setValue(scrollbar.maximum())
        self.app.processEvents()
        self.assertEqual(title.mapTo(panel, QPoint()).y(), original_y)

    def test_switch_and_parameter_edits_keep_existing_signal_contracts(self):
        panel = self.detector_panel()
        checkbox = panel.detect_config_panel.keep_existing_checker
        with patch.object(pcfg.module, 'keep_exist_textlines', False):
            checkbox.setFocus()
            QTest.keyClick(checkbox, Qt.Key.Key_Space)
            self.assertTrue(pcfg.module.keep_exist_textlines)
            panel.focusOnOCR()
            panel.focusOnDetect()
            self.assertTrue(checkbox.isChecked())
        changes = []
        panel.detect_config_panel.paramwidget_edited.connect(lambda key, value: changes.append((key, value)))
        editor = next(w for w in panel.detect_config_panel.findChildren(ParamLineEditor)
                      if w.param_key == 'font size multiplier')
        editor.setText('1.25')
        self.assertEqual(changes[-1], ('font size multiplier', {'content': '1.25'}))

    def test_loading_language_options_does_not_expand_aligned_controls(self):
        panel = ConfigPanel()
        panel.show()
        panel.focusOnTranslator()
        translator = panel.trans_config_panel
        translator.source_combobox.addItems(['A very long translated language name ' * 3])
        translator.target_combobox.addItems(['English', 'Korean'])
        self.app.processEvents()
        self.assertEqual(translator.source_combobox.width(), translator.module_combobox.width())
        self.assertEqual(translator.target_combobox.width(), translator.module_combobox.width())
        self.assertEqual(panel.configContent.horizontalScrollBar().maximum(), 0)

    def test_inpaint_selectors_remain_in_canvas_when_settings_opens_and_closes(self):
        panel = ConfigPanel()
        source = panel.inpaint_config_panel
        source.get_valid_module_keys = lambda: ['first', 'second']
        source.addModulesParamWidgets({'first': {}, 'second': {}})
        for view_type in (InpaintPanel, RectPanel):
            with self.subTest(view=view_type.__name__):
                canvas_tools = view_type(source)
                canvas_tools.show()
                self.app.processEvents()
                combo = canvas_tools.inpaint_layout.itemAt(1).widget()
                panel.show()
                panel.focusOnInpaint()
                self.app.processEvents()
                self.assertIs(combo.parentWidget(), canvas_tools)
                self.assertTrue(combo.isVisible())
                self.assertIsNot(combo, source.module_combobox)
                source.setModule('second')
                self.assertEqual(combo.currentText(), 'second')
                combo.setFocus()
                QTest.keyClick(combo, Qt.Key.Key_Up)
                self.assertEqual(source.module_combobox.currentText(), 'first')
                panel.close()
                self.app.processEvents()
                self.assertTrue(combo.isVisible())
                self.assertGreaterEqual(canvas_tools.inpaint_layout.indexOf(combo), 0)
                canvas_tools.close()

    def test_grouped_checkboxes_are_not_clipped_in_a_small_window(self):
        panel = ConfigPanel()
        detector = panel.detect_config_panel
        detector.get_valid_module_keys = lambda: ['grouped']
        detector.addModulesParamWidgets({'grouped': {'classes': {
            'type': 'check_group', 'value': {
                name: True for name in ('text', 'bubble', 'changfangtiao', 'rectangle', 'circle', 'other')},
        }}})
        panel.resize(800, 700)
        panel.show()
        panel.focusOnDetect()
        self.app.processEvents()
        group = detector.findChild(ParamCheckGroup)
        for checkbox in group.findChildren(QCheckBox):
            self.assertGreaterEqual(checkbox.width(), checkbox.minimumSizeHint().width())

    def test_config_panel_is_a_compact_modeless_tool_window(self):
        owner = QWidget()
        panel = ConfigPanel(owner)

        panel.show()
        self.app.processEvents()

        self.assertTrue(panel.isWindow())
        self.assertTrue(panel.windowFlags() & Qt.WindowType.Tool)
        self.assertTrue(panel.isVisible())
        self.assertLessEqual(panel.width(), 960)
        self.assertLessEqual(panel.height(), 760)
        self.assertEqual(panel.windowModality(), Qt.WindowModality.NonModal)

    def test_config_panel_uses_korean_window_title_immediately(self):
        owner = QWidget()
        with patch.object(pcfg, 'display_lang', 'ko_KR'):
            panel = ConfigPanel(owner)

        self.assertEqual(panel.windowTitle(), '설정')

    def test_setup_config_ui_keeps_workspace_visible_and_opens_panel(self):
        owner = QWidget()
        workspace_stack = QStackedWidget(owner)
        workspace_stack.addWidget(QWidget())
        workspace_stack.addWidget(QWidget())
        workspace_stack.setCurrentIndex(0)
        panel = ConfigPanel(owner)
        harness = SimpleNamespace(
            centralStackWidget=workspace_stack,
            configPanel=panel,
        )

        MainWindow.setupConfigUI(harness)
        self.app.processEvents()

        self.assertEqual(workspace_stack.currentIndex(), 0)
        self.assertTrue(panel.isVisible())

    def test_config_panel_opens_centered_over_its_owner(self):
        owner = QWidget()
        owner.resize(1200, 900)
        owner.move(100, 50)
        owner.show()
        panel = ConfigPanel(owner)

        panel.show()
        self.app.processEvents()

        owner_center = owner.frameGeometry().center()
        panel_center = panel.frameGeometry().center()
        self.assertLessEqual(abs(owner_center.x() - panel_center.x()), 5)
        self.assertLessEqual(abs(owner_center.y() - panel_center.y()), 5)

    def test_launcher_and_panel_visibility_stay_in_sync(self):
        owner = QWidget()
        panel = ConfigPanel(owner)
        launcher = StateChecker('config', uncheckable=True)
        launcher.show()
        self.app.processEvents()

        panel.bindVisibilityToggle(launcher)

        QTest.mouseClick(launcher, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(launcher.isChecked())
        self.assertTrue(panel.isVisible())

        QTest.mouseClick(launcher, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertFalse(launcher.isChecked())
        self.assertFalse(panel.isVisible())

        launcher.setChecked(True)
        self.app.processEvents()
        panel.close()
        self.app.processEvents()
        self.assertFalse(launcher.isChecked())

    def test_existing_settings_navigation_structure_is_preserved(self):
        owner = QWidget()
        panel = ConfigPanel(owner)
        model = panel.configTable.model()

        self.assertEqual(model.item(0, 0).text(), 'DL Module')
        self.assertEqual(
            [model.item(0, 0).child(row).text() for row in range(4)],
            ['Text Detection', 'OCR', 'Inpaint', 'Translator'],
        )
        self.assertEqual(model.item(1, 0).text(), 'General')
        self.assertEqual(
            [model.item(1, 0).child(row).text() for row in range(4)],
            ['Startup', 'Typesetting', 'Save', 'SalaDict'],
        )


if __name__ == "__main__":
    unittest.main()
