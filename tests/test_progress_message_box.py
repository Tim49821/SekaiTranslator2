import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QPushButton, QWidget

from ui.custom_widget.message import ImgtransProgressMessageBox, ProgressMessageBox


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class ProgressMessageBoxTest(unittest.TestCase):
    def test_imgtrans_progress_dialog_has_one_stop_button(self):
        app = ensure_app()
        dialog = ImgtransProgressMessageBox()

        buttons = dialog.findChildren(QPushButton)

        self.assertEqual(buttons, [dialog.stop_button])
        dialog.close()
        app.processEvents()

    def test_child_progress_dialog_stays_hidden_until_explicitly_shown(self):
        app = ensure_app()
        parent = QWidget()
        dialog = ProgressMessageBox("Preparing module: ", True, parent)

        parent.show()
        app.processEvents()

        self.assertTrue(dialog.isHidden())
        self.assertFalse(dialog.isVisible())

        dialog.show_fitted()
        app.processEvents()

        self.assertTrue(dialog.isVisible())

        dialog.hide()
        parent.close()

    def test_show_fitted_restores_height_when_dialog_was_already_visible(self):
        app = ensure_app()
        parent = QWidget()
        dialog = ProgressMessageBox("Installing packages: ", True, parent)

        parent.show()
        dialog.show()
        app.processEvents()
        dialog.resize(100, 30)

        dialog.show_fitted()
        app.processEvents()

        self.assertGreater(dialog.height(), 30)
        self.assertGreaterEqual(dialog.height(), dialog.sizeHint().height())
        self.assertTrue(dialog.task_progress_bar.isVisible())
        self.assertTrue(dialog.stop_button.isVisible())

        dialog.hide()
        parent.close()


if __name__ == "__main__":
    unittest.main()
