import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from qtpy.QtCore import QPointF, QSize, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPen
try:
    from qtpy.QtWidgets import QApplication, QUndoCommand, QWidget
except ImportError:
    from qtpy.QtGui import QUndoCommand
    from qtpy.QtWidgets import QApplication, QWidget

from ui.canvas import Canvas
from ui.drawing_commands import InpaintUndoCommand, StrokeItemUndoCommand
from ui.drawingpanel import DrawingPanel
from ui.image_edit import DrawingLayer, ImageEditMode, StrokeImgItem
from utils.config import DrawPanelConfig


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FakeProject:
    def __init__(self):
        self.inpainted_array = np.zeros((6, 6, 3), dtype=np.uint8)
        self.mask_array = np.zeros((6, 6), dtype=np.uint8)


class FakeCanvas:
    def __init__(self):
        self.imgtrans_proj = FakeProject()
        self.update_count = 0

    def updateLayers(self):
        self.update_count += 1


class CounterCommand(QUndoCommand):
    def __init__(self):
        super().__init__()
        self.value = 0

    def redo(self):
        self.value += 1

    def undo(self):
        self.value -= 1


class FakeInpainterPanel:
    def __init__(self):
        self.module_combobox = QWidget()


class PaintModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = ensure_app()

    def test_stroke_clip_returns_none_for_empty_alpha(self):
        pen = QPen(QColor(0, 0, 0, 255), 8)
        stroke = StrokeImgItem(pen, QPointF(20, 20), QSize(64, 64))
        stroke.finishPainting()
        stroke._img.fill(Qt.GlobalColor.transparent)

        rect, mask, qimg = stroke.clip(mask_only=True)

        self.assertIsNone(rect)
        self.assertIsNone(mask)
        self.assertIsNone(qimg)

    def test_stroke_clip_returns_mask_for_drawn_stroke(self):
        pen = QPen(QColor(0, 0, 0, 255), 8)
        stroke = StrokeImgItem(pen, QPointF(20, 20), QSize(64, 64))
        stroke.lineTo(QPointF(35, 20))
        stroke.finishPainting()

        rect, mask, qimg = stroke.clip(mask_only=True)

        self.assertIsNotNone(rect)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(qimg)
        self.assertGreater(mask.sum(), 0)

    def test_inpaint_undo_command_restores_image_and_mask_rect(self):
        canvas = FakeCanvas()
        canvas.imgtrans_proj.inpainted_array[1:4, 1:4] = 7
        canvas.imgtrans_proj.mask_array[1:4, 1:4] = 10
        redo_img = np.full((3, 3, 3), 99, dtype=np.uint8)
        redo_mask = np.full((3, 3), 255, dtype=np.uint8)

        command = InpaintUndoCommand(canvas, redo_img, redo_mask, [1, 1, 4, 4])

        command.redo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[1:4, 1:4], redo_img)
        np.testing.assert_array_equal(canvas.imgtrans_proj.mask_array[1:4, 1:4], redo_mask)

        command.undo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[1:4, 1:4], np.full((3, 3, 3), 7, dtype=np.uint8))
        np.testing.assert_array_equal(canvas.imgtrans_proj.mask_array[1:4, 1:4], np.full((3, 3), 10, dtype=np.uint8))
        self.assertEqual(canvas.update_count, 2)

    def test_stroke_item_undo_command_updates_layer_without_scene(self):
        layer = DrawingLayer()
        qimg = QImage(4, 4, QImage.Format.Format_ARGB32)
        qimg.fill(QColor(1, 2, 3, 255))

        command = StrokeItemUndoCommand(layer, (2, 3, 4, 4), qimg, erasing=True)
        command.redo()

        self.assertIn(command.key, layer.qimg_dict)
        self.assertEqual(layer.drawing_items_info[command.key]["pos"], [2, 3])
        self.assertEqual(
            layer.drawing_items_info[command.key]["compose"],
            QPainter.CompositionMode.CompositionMode_DestinationOut,
        )

        command.undo()

        self.assertNotIn(command.key, layer.qimg_dict)

    def test_canvas_undo_redo_only_changes_counts_when_available(self):
        canvas = Canvas()
        canvas.editor_index = 0
        command = CounterCommand()

        canvas.undo()
        canvas.redo()
        self.assertEqual(canvas.num_pushed_drawstep, 0)
        self.assertEqual(command.value, 0)

        canvas.push_draw_command(command)
        self.assertEqual(canvas.num_pushed_drawstep, 1)
        self.assertEqual(command.value, 1)

        canvas.undo()
        self.assertEqual(canvas.num_pushed_drawstep, 0)
        self.assertEqual(command.value, 0)

        canvas.undo()
        self.assertEqual(canvas.num_pushed_drawstep, 0)
        self.assertEqual(command.value, 0)

        canvas.redo()
        self.assertEqual(canvas.num_pushed_drawstep, 1)
        self.assertEqual(command.value, 1)

        canvas.redo()
        self.assertEqual(canvas.num_pushed_drawstep, 1)
        self.assertEqual(command.value, 1)

    def test_eraser_tool_is_selectable_painting_tool(self):
        canvas = Canvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        config = DrawPanelConfig(
            erasertool_width=17,
            erasertool_shape=1,
            current_tool=ImageEditMode.EraserTool,
        )

        panel.set_config(config)

        self.assertIs(panel.currentTool, panel.eraserTool)
        self.assertEqual(canvas.image_edit_mode, ImageEditMode.EraserTool)
        self.assertTrue(canvas.painting)
        self.assertEqual(canvas.erasing_pen.widthF(), 17)
        self.assertEqual(canvas.painting_shape, 1)


if __name__ == "__main__":
    unittest.main()
