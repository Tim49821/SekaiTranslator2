import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPen
try:
    from qtpy.QtWidgets import QApplication, QComboBox, QUndoCommand, QWidget
except ImportError:
    from qtpy.QtGui import QUndoCommand
    from qtpy.QtWidgets import QApplication, QComboBox, QWidget

from ui.canvas import Canvas
from ui.drawing_commands import InpaintHardResetCommand, InpaintUndoCommand, StrokeItemUndoCommand
from ui.drawingpanel import DrawingPanel, _expand_context_rect
from ui.funcmaps import (
    MASKSEG_EXISTING_MASK,
    MASKSEG_METHOD_1,
    MASKSEG_METHOD_2,
    MASKSEG_METHOD_3,
    get_maskseg_method,
)
from ui.image_edit import DrawingLayer, ImageEditMode, PenShape, StrokeImgItem
from ui.module_manager import InpaintThread, ModuleManager
from utils.config import DrawPanelConfig, pcfg
from utils.textblock_mask import canny_flood_natural, existing_mask


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FakeProject:
    def __init__(self):
        self.inpainted_array = np.zeros((6, 6, 3), dtype=np.uint8)
        self.mask_array = np.zeros((6, 6), dtype=np.uint8)


class FakeTextProject:
    def __init__(self):
        self.img_array = np.zeros((200, 300, 3), dtype=np.uint8)
        self.inpainted_array = np.copy(self.img_array)
        self.mask_array = np.zeros((200, 300), dtype=np.uint8)

    @property
    def img_valid(self):
        return True

    @property
    def inpainted_valid(self):
        return True

    @property
    def mask_valid(self):
        return self.mask_array is not None


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
        self.module_combobox = QComboBox()


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

    def test_method3_combo_keeps_existing_mask_config_compatible(self):
        canvas = Canvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        combo = panel.rectPanel.methodComboBox

        self.assertEqual(
            [combo.itemData(index) for index in range(combo.count())],
            [MASKSEG_METHOD_1, MASKSEG_METHOD_2, MASKSEG_METHOD_3, MASKSEG_EXISTING_MASK],
        )
        panel.set_config(DrawPanelConfig(rectool_method=MASKSEG_EXISTING_MASK))
        self.assertEqual(combo.currentData(), MASKSEG_EXISTING_MASK)
        self.assertIs(get_maskseg_method(MASKSEG_EXISTING_MASK), existing_mask)

        panel.set_config(DrawPanelConfig(rectool_method=MASKSEG_METHOD_3))
        self.assertEqual(combo.currentData(), MASKSEG_METHOD_3)
        self.assertIs(get_maskseg_method(MASKSEG_METHOD_3), canny_flood_natural)

    def test_method3_bypasses_flat_fill_and_expands_model_context(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()
        yy, xx = np.indices(canvas.imgtrans_proj.inpainted_array.shape[:2])
        canvas.imgtrans_proj.inpainted_array[..., 0] = xx % 251
        canvas.imgtrans_proj.inpainted_array[..., 1] = yy % 251
        canvas.imgtrans_proj.inpainted_array[..., 2] = (xx + yy) % 251
        original_page = canvas.imgtrans_proj.inpainted_array.copy()
        panel = DrawingPanel(canvas, FakeInpainterPanel())

        selected_rect = [100, 60, 140, 90]
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[10:20, 12:28] = 255
        calls = []
        panel.runInpaint = lambda inpaint_dict=None: calls.append(inpaint_dict)
        inpaint_dict = {
            "img": original_page[60:90, 100:140].copy(),
            "mask": mask,
            "inpaint_rect": selected_rect,
            "need_inpaint": False,
            "bground_rgb": np.array([255, 255, 255], dtype=np.uint8),
            "ballon_mask": np.full(mask.shape, 255, dtype=np.uint8),
            "force_inpaint": True,
            "context_ratio": 2.5,
            "feather_radius": 2,
        }

        panel.inpaintRect(inpaint_dict)

        self.assertEqual(len(calls), 1)
        prepared = calls[0]
        ex1, ey1, ex2, ey2 = prepared["inpaint_rect"]
        self.assertLess(ex1, selected_rect[0])
        self.assertLess(ey1, selected_rect[1])
        self.assertGreater(ex2, selected_rect[2])
        self.assertGreater(ey2, selected_rect[3])
        self.assertEqual(prepared["img"].shape[:2], prepared["mask"].shape)
        offset_x = selected_rect[0] - ex1
        offset_y = selected_rect[1] - ey1
        np.testing.assert_array_equal(
            prepared["mask"][offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]],
            mask,
        )
        self.assertEqual(np.count_nonzero(prepared["mask"]), np.count_nonzero(mask))
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array, original_page)

        edge_rect = _expand_context_rect([0, 0, 40, 30], 300, 200, 2.5)
        self.assertEqual(edge_rect[:2], [0, 0])
        self.assertGreater(edge_rect[2], 40)
        self.assertGreater(edge_rect[3], 30)

    def test_method3_finished_result_uses_soft_edge_and_tracks_effective_mask(self):
        canvas = Canvas()
        canvas.editor_index = 0
        canvas.imgtrans_proj = FakeTextProject()
        canvas.updateCanvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.paint_busy = True
        canvas.paint_busy = True

        rect = [20, 20, 40, 40]
        original = np.zeros((20, 20, 3), dtype=np.uint8)
        candidate = np.full_like(original, 200)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[7:13, 7:13] = 255
        panel.on_inpaint_finished({
            "img": original,
            "inpainted": candidate,
            "mask": mask,
            "inpaint_rect": rect,
            "feather_radius": 2,
        })

        result = canvas.imgtrans_proj.inpainted_array[20:40, 20:40]
        stored_mask = canvas.imgtrans_proj.mask_array[20:40, 20:40]
        self.assertTrue(np.all(result[mask > 0] == 200))
        self.assertTrue(np.all(result[stored_mask == 0] == 0))
        margin = (stored_mask > 0) & (mask == 0)
        margin_values = result[..., 0][margin]
        self.assertTrue(np.any((margin_values > 0) & (margin_values < 200)))

    def test_canvas_inpaint_prepare_failure_emits_failure_signal(self):
        manager = ModuleManager(None)
        failures = []
        callbacks = {}
        manager.inpaint_thread = SimpleNamespace(
            inpainting=False,
            inpaint_failed=SimpleNamespace(emit=lambda: failures.append(True)),
        )
        manager._prepare_modules_then = lambda required, on_success, on_failure=None: callbacks.update(
            success=on_success,
            failure=on_failure,
        )

        manager.canvas_inpaint({"img": np.zeros((2, 2, 3)), "mask": np.zeros((2, 2))})
        callbacks["failure"]()

        self.assertEqual(failures, [True])
        self.assertFalse(manager.run_canvas_inpaint)

    def test_inpaint_thread_preserves_method3_blend_metadata(self):
        calls = []
        finished = []

        class FakeInpainter:
            def inpaint(self, img, mask):
                calls.append((img.copy(), mask.copy()))
                return np.full_like(img, 123)

        fake_thread = SimpleNamespace(
            inpainter=FakeInpainter(),
            inpainting=False,
            stop_requested=False,
            finish_inpaint=SimpleNamespace(emit=lambda payload: finished.append(payload)),
            inpaint_failed=SimpleNamespace(emit=lambda: None),
            tr=lambda text: text,
        )
        img = np.zeros((6, 6, 3), dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2:4, 2:4] = 255

        InpaintThread._inpaint(
            fake_thread,
            img,
            mask,
            inpaint_rect=[0, 0, 6, 6],
            force_inpaint=True,
            feather_radius=2,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["force_inpaint"])
        self.assertEqual(finished[0]["feather_radius"], 2)
        self.assertTrue(np.all(finished[0]["inpainted"] == 123))

    def test_page_change_invalidates_pending_canvas_inpaint(self):
        manager = ModuleManager(None)
        callbacks = {}
        starts = []
        manager.inpaint_thread = SimpleNamespace(
            inpainting=False,
            inpaint_failed=SimpleNamespace(emit=lambda: None),
            requestStop=lambda: None,
        )
        manager.imgtrans_thread = SimpleNamespace(isRunning=lambda: False)
        manager.inpaint = lambda **kwargs: starts.append(kwargs)
        manager._prepare_modules_then = lambda required, on_success, on_failure=None: callbacks.update(
            success=on_success,
            failure=on_failure,
        )

        manager.canvas_inpaint({"img": np.zeros((2, 2, 3)), "mask": np.zeros((2, 2))})
        manager.handle_page_changed()
        callbacks["success"]()

        self.assertEqual(starts, [])
        self.assertFalse(manager.run_canvas_inpaint)

    def test_page_change_discards_queued_result_from_previous_page(self):
        manager = ModuleManager(None)
        emitted = []
        manager.canvas_inpaint_finished.connect(lambda payload: emitted.append(payload))
        manager.inpaint_thread = SimpleNamespace(
            inpainting=False,
            requestStop=lambda: None,
        )
        manager.imgtrans_thread = SimpleNamespace(isRunning=lambda: False)
        manager._canvas_inpaint_request_id = 7
        manager.run_canvas_inpaint = True

        manager.handle_page_changed()
        manager.on_finish_inpaint({
            "inpainted": np.zeros((2, 2, 3), dtype=np.uint8),
            "_canvas_inpaint_request_id": 7,
        })

        self.assertEqual(emitted, [])
        self.assertFalse(manager.run_canvas_inpaint)

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

    def test_inpaint_undo_command_matches_redo_image_channels(self):
        canvas = FakeCanvas()
        canvas.imgtrans_proj.inpainted_array[1:4, 1:4] = 7
        canvas.imgtrans_proj.mask_array[1:4, 1:4] = 10
        redo_img = np.zeros((3, 3, 4), dtype=np.uint8)
        redo_img[:, :] = [11, 12, 13, 200]
        redo_mask = np.full((3, 3), 255, dtype=np.uint8)

        command = InpaintUndoCommand(canvas, redo_img, redo_mask, [1, 1, 4, 4])
        command.redo()

        np.testing.assert_array_equal(
            canvas.imgtrans_proj.inpainted_array[1:4, 1:4],
            np.full((3, 3, 3), [11, 12, 13], dtype=np.uint8),
        )
        np.testing.assert_array_equal(canvas.imgtrans_proj.mask_array[1:4, 1:4], redo_mask)

    def test_inpaint_hard_reset_command_restores_whole_page_and_mask(self):
        canvas = FakeCanvas()
        canvas.imgtrans_proj.img_array = np.full((6, 6, 3), 4, dtype=np.uint8)
        canvas.imgtrans_proj.inpainted_array[1:5, 1:5] = 9
        canvas.imgtrans_proj.mask_array[1:5, 1:5] = 255

        command = InpaintHardResetCommand(canvas)

        command.redo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array, canvas.imgtrans_proj.img_array)
        self.assertEqual(canvas.imgtrans_proj.mask_array.sum(), 0)

        command.undo()
        self.assertEqual(canvas.imgtrans_proj.inpainted_array[2, 2, 0], 9)
        self.assertEqual(canvas.imgtrans_proj.mask_array[2, 2], 255)
        self.assertEqual(canvas.update_count, 2)

    def test_manual_rectangle_selection_replaces_preview_with_undo(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.on_use_recttool()
        panel.rectPanel.autoChecker.setChecked(False)
        mask_values = iter((40, 80))

        def segment_rect(image, mask=None):
            value = next(mask_values)
            raw_mask = np.full(image.shape[:2], value, dtype=np.uint8)
            balloon_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return raw_mask, balloon_mask, {
                "bground_rgb": np.array([10, 20, 30], dtype=np.uint8),
                "need_inpaint": True,
            }

        with patch("ui.drawingpanel.get_maskseg_method", return_value=segment_rect):
            panel.on_end_create_rect(QRectF(10, 15, 30, 25), 0)
            self.assertEqual(canvas.image_edit_mode, ImageEditMode.RectTool)
            self.assertEqual(panel.rect_inpaint_dict["inpaint_rect"], [10, 15, 40, 40])
            self.assertEqual(panel.inpaint_mask_item.pos(), QPointF(10, 15))
            first_preview = panel.inpaint_mask_item.pixmap().toImage().copy()

            canvas.startCreateTextblock(QPointF(60, 65), hide_control=True)
            self.assertIsNone(panel.inpaint_mask_item.scene())
            panel.on_end_create_rect(QRectF(60, 65, 20, 15), 0)
            self.assertEqual(panel.rect_inpaint_dict["inpaint_rect"], [60, 65, 80, 80])
            self.assertTrue(np.all(panel.inpaint_mask_array == 80))
            second_preview = panel.inpaint_mask_item.pixmap().toImage().copy()
            self.assertNotEqual(second_preview, first_preview)

        canvas.undo()
        self.assertEqual(panel.rect_inpaint_dict["inpaint_rect"], [10, 15, 40, 40])
        self.assertTrue(np.all(panel.inpaint_mask_array == 40))
        self.assertEqual(panel.inpaint_mask_item.pos(), QPointF(10, 15))
        self.assertIs(panel.inpaint_mask_item.scene(), canvas)
        self.assertEqual(panel.inpaint_mask_item.pixmap().toImage(), first_preview)

        canvas.redo()
        self.assertEqual(panel.rect_inpaint_dict["inpaint_rect"], [60, 65, 80, 80])
        self.assertTrue(np.all(panel.inpaint_mask_array == 80))
        self.assertEqual(panel.inpaint_mask_item.pos(), QPointF(60, 65))
        self.assertEqual(panel.inpaint_mask_item.pixmap().toImage(), second_preview)

    def test_invalid_replacement_gesture_restores_previous_manual_preview(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.on_use_recttool()
        panel.rectPanel.autoChecker.setChecked(False)

        def segment_rect(image, mask=None):
            raw_mask = np.full(image.shape[:2], 60, dtype=np.uint8)
            balloon_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return raw_mask, balloon_mask, {
                "bground_rgb": np.array([10, 20, 30], dtype=np.uint8),
                "need_inpaint": True,
            }

        with patch("ui.drawingpanel.get_maskseg_method", return_value=segment_rect):
            panel.on_end_create_rect(QRectF(10, 15, 30, 25), 0)

        canvas.startCreateTextblock(QPointF(60, 65), hide_control=True)
        self.assertIsNone(panel.inpaint_mask_item.scene())
        panel.on_end_create_rect(QRectF(60, 65, 1, 1), 0)

        self.assertEqual(panel.rect_inpaint_dict["inpaint_rect"], [10, 15, 40, 40])
        self.assertEqual(panel.inpaint_mask_item.pos(), QPointF(10, 15))
        self.assertIs(panel.inpaint_mask_item.scene(), canvas)

    def test_applied_manual_rectangle_clears_preview_and_keeps_image_undoable(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.on_use_recttool()
        panel.rectPanel.autoChecker.setChecked(False)
        old_check_need_inpaint = pcfg.module.check_need_inpaint
        pcfg.module.check_need_inpaint = True

        def segment_rect(image, mask=None):
            raw_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            balloon_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return raw_mask, balloon_mask, {
                "bground_rgb": np.array([70, 80, 90], dtype=np.uint8),
                "need_inpaint": False,
            }

        try:
            with patch("ui.drawingpanel.get_maskseg_method", return_value=segment_rect):
                panel.on_end_create_rect(QRectF(10, 15, 30, 25), 0)
            panel.on_rect_inpaintbtn_clicked()

            self.assertIsNone(panel.rect_inpaint_dict)
            self.assertIsNone(panel.inpaint_mask_item.scene())
            np.testing.assert_array_equal(
                canvas.imgtrans_proj.inpainted_array[15:40, 10:40],
                np.full((25, 30, 3), [70, 80, 90], dtype=np.uint8),
            )

            canvas.undo()
            np.testing.assert_array_equal(
                canvas.imgtrans_proj.inpainted_array[15:40, 10:40],
                np.zeros((25, 30, 3), dtype=np.uint8),
            )
            self.assertEqual(canvas.imgtrans_proj.mask_array.sum(), 0)
        finally:
            pcfg.module.check_need_inpaint = old_check_need_inpaint

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

    def test_canvas_tracks_manual_textblock_availability(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()

        state = canvas.manual_textblock_state()
        self.assertFalse(state["can_quick_create"])
        self.assertFalse(state["can_drag_create"])
        self.assertFalse(state["can_delete"])

        canvas.editor_index = 1
        state = canvas.manual_textblock_state()
        self.assertTrue(state["can_quick_create"])
        self.assertFalse(state["can_drag_create"])

        canvas.setTextBlockMode(True)
        canvas.txtblkShapeControl.blk_item = object()
        state = canvas.manual_textblock_state()
        self.assertTrue(state["can_drag_create"])
        self.assertTrue(state["can_delete"])
        self.assertTrue(state["has_active_textblock"])

    def test_create_textblock_at_emits_default_rect_in_image_coordinates(self):
        canvas = Canvas()
        canvas.imgtrans_proj = FakeTextProject()
        canvas.editor_index = 1
        canvas.scale_factor = 2.0
        emitted_rects = []
        canvas.end_create_textblock.connect(lambda rect: emitted_rects.append(rect))

        created = canvas.create_textblock_at(QPointF(1000, 1000))

        self.assertTrue(created)
        self.assertEqual(len(emitted_rects), 1)
        rect = emitted_rects[0]
        self.assertEqual(rect.x(), 60)
        self.assertEqual(rect.y(), 110)
        self.assertEqual(rect.width(), 240)
        self.assertEqual(rect.height(), 90)

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

    def test_restore_tool_is_selectable_painting_tool(self):
        canvas = Canvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        config = DrawPanelConfig(
            restoretool_width=19,
            restoretool_shape=1,
            current_tool=ImageEditMode.RestoreTool,
        )

        panel.set_config(config)

        self.assertIs(panel.currentTool, panel.restoreTool)
        self.assertEqual(canvas.image_edit_mode, ImageEditMode.RestoreTool)
        self.assertTrue(canvas.painting)
        self.assertEqual(canvas.painting_pen.widthF(), 19)
        self.assertEqual(canvas.painting_shape, 1)
        self.assertIs(panel.inpaintHardResetBtn.parent(), panel.restoreConfigPanel)

    def test_draw_tool_tooltips_include_name_and_shortcut(self):
        canvas = Canvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())

        tooltips = {
            "hand": (panel.handTool, "H", "Hand (H)"),
            "inpaint": (panel.inpaintTool, "J", "Inpaint (J)"),
            "restore": (panel.restoreTool, "O", "Restore (O)"),
            "pen": (panel.penTool, "B", "Pen (B)"),
            "eraser": (panel.eraserTool, "E", "Eraser (E)"),
            "rect": (panel.rectTool, "R", "Rectangle (R)"),
        }

        for tool_name, (tool, shortcut, expected_tip) in tooltips.items():
            panel.setShortcutTip(tool_name, shortcut)
            self.assertEqual(tool.toolTip(), expected_tip)

    def test_draw_tool_tooltips_use_korean_fallback(self):
        original_lang = pcfg.display_lang
        try:
            pcfg.display_lang = "ko_KR"
            canvas = Canvas()
            panel = DrawingPanel(canvas, FakeInpainterPanel())

            panel.setShortcutTip("hand", "H")
            panel.setShortcutTip("pen", "B")

            self.assertEqual(panel.handTool.toolTip(), "손 도구 (H)")
            self.assertEqual(panel.penTool.toolTip(), "펜 (B)")
        finally:
            pcfg.display_lang = original_lang

    def test_restore_tool_restores_inpainted_pixels_to_original(self):
        canvas = Canvas()
        canvas.editor_index = 0
        canvas.imgtrans_proj = FakeTextProject()
        canvas.imgtrans_proj.img_array[:, :] = [10, 20, 30]
        canvas.imgtrans_proj.inpainted_array = np.copy(canvas.imgtrans_proj.img_array)
        canvas.imgtrans_proj.inpainted_array[8:16, 8:16] = [90, 91, 92]
        canvas.imgtrans_proj.mask_array[8:16, 8:16] = 255
        canvas.updateCanvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.set_config(DrawPanelConfig(
            restoretool_width=5,
            restoretool_shape=PenShape.Rectangle,
            current_tool=ImageEditMode.RestoreTool,
        ))
        stroke = StrokeImgItem(
            canvas.painting_pen,
            QPointF(12, 12),
            canvas.img_window_size(),
            shape=PenShape.Rectangle,
        )
        stroke.setParentItem(canvas.baseLayer)

        panel.on_finish_painting(stroke)

        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[12, 12], [10, 20, 30])
        self.assertEqual(canvas.imgtrans_proj.mask_array[12, 12], 0)
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[8, 8], [90, 91, 92])
        self.assertEqual(canvas.imgtrans_proj.mask_array[8, 8], 255)

        canvas.undo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[12, 12], [90, 91, 92])
        self.assertEqual(canvas.imgtrans_proj.mask_array[12, 12], 255)

        canvas.redo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[12, 12], [10, 20, 30])
        self.assertEqual(canvas.imgtrans_proj.mask_array[12, 12], 0)

    def test_restore_tool_handles_rgba_original_with_rgb_inpainted_image(self):
        canvas = Canvas()
        canvas.editor_index = 0
        canvas.imgtrans_proj = FakeTextProject()
        canvas.imgtrans_proj.img_array = np.zeros((200, 300, 4), dtype=np.uint8)
        canvas.imgtrans_proj.img_array[:, :] = [10, 20, 30, 180]
        canvas.imgtrans_proj.inpainted_array = canvas.imgtrans_proj.img_array[..., :3].copy()
        canvas.imgtrans_proj.inpainted_array[8:16, 8:16] = [90, 91, 92]
        canvas.imgtrans_proj.mask_array[8:16, 8:16] = 255
        canvas.updateCanvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())
        panel.set_config(DrawPanelConfig(
            restoretool_width=5,
            restoretool_shape=PenShape.Rectangle,
            current_tool=ImageEditMode.RestoreTool,
        ))
        stroke = StrokeImgItem(
            canvas.painting_pen,
            QPointF(12, 12),
            canvas.img_window_size(),
            shape=PenShape.Rectangle,
        )
        stroke.setParentItem(canvas.baseLayer)

        panel.on_finish_painting(stroke)

        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[12, 12], [10, 20, 30])
        self.assertEqual(canvas.imgtrans_proj.mask_array[12, 12], 0)
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[8, 8], [90, 91, 92])
        self.assertEqual(canvas.imgtrans_proj.mask_array[8, 8], 255)

    def test_hard_reset_inpaint_is_undoable_from_panel(self):
        canvas = Canvas()
        canvas.editor_index = 0
        canvas.imgtrans_proj = FakeTextProject()
        canvas.imgtrans_proj.img_array[:, :] = [4, 5, 6]
        canvas.imgtrans_proj.inpainted_array[8:16, 8:16] = [90, 91, 92]
        canvas.imgtrans_proj.mask_array[8:16, 8:16] = 255
        canvas.updateCanvas()
        panel = DrawingPanel(canvas, FakeInpainterPanel())

        self.assertTrue(panel.hardResetInpaint())
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array, canvas.imgtrans_proj.img_array)
        self.assertEqual(canvas.imgtrans_proj.mask_array.sum(), 0)

        canvas.undo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array[12, 12], [90, 91, 92])
        self.assertEqual(canvas.imgtrans_proj.mask_array[12, 12], 255)

        canvas.redo()
        np.testing.assert_array_equal(canvas.imgtrans_proj.inpainted_array, canvas.imgtrans_proj.img_array)
        self.assertEqual(canvas.imgtrans_proj.mask_array.sum(), 0)


if __name__ == "__main__":
    unittest.main()
