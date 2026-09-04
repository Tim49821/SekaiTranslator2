from qtpy.QtCore import Signal, Qt, QPointF, QSize, QSizeF, QLineF, QRectF
from qtpy.QtWidgets import QGridLayout, QPushButton, QComboBox, QSizePolicy, QBoxLayout, QCheckBox, QHBoxLayout, QGraphicsView, QStackedWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem, QGraphicsEllipseItem
from qtpy.QtGui import QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics

from typing import Union, Tuple, List
import copy
import numpy as np
import cv2

from utils.imgproc_utils import enlarge_window, restore_masked_pixels
from utils.textblock_mask import canny_flood, connected_canny_flood, feather_inpaint_result
from utils.logger import logger
from utils.config import pcfg
from .funcmaps import (
    MASKSEG_EXISTING_MASK,
    MASKSEG_METHOD_1,
    MASKSEG_METHOD_2,
    MASKSEG_METHOD_3,
    get_maskseg_method,
)
from .module_manager import ModuleManager
from .image_edit import ImageEditMode, PenShape, PixmapItem, StrokeImgItem
from .configpanel import InpaintConfigPanel
from .custom_widget import Widget, SeparatorWidget, PaintQSlider, ColorPickerLabel, ConfigComboBox
from .canvas import Canvas
from .misc import ndarray2pixmap
from utils.config import DrawPanelConfig, pcfg
from utils.shared import CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.logger import logger as LOGGER
from .drawing_commands import InpaintHardResetCommand, InpaintUndoCommand, RectInpaintSelectionCommand, StrokeItemUndoCommand

INPAINT_BRUSH_COLOR = QColor(127, 0, 127, 127)
RESTORE_BRUSH_COLOR = QColor(30, 120, 210, 127)
MAX_PEN_SIZE = 1000
MIN_PEN_SIZE = 1
TOOLNAME_POINT_SIZE = 13
MAX_CURSOR_PIXMAP_SIZE = 256


def _expand_context_rect(rect, im_w: int, im_h: int, area_ratio: float):
    """Expand toward available space instead of requiring symmetric margins."""
    x1, y1, x2, y2 = [int(value) for value in rect]
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return [x1, y1, x2, y2]

    scale = float(np.sqrt(max(area_ratio, 1.0)))
    target_w = min(im_w, max(width, int(np.ceil(width * scale))))
    target_h = min(im_h, max(height, int(np.ceil(height * scale))))
    expanded_x1 = max(0, min(x1 - (target_w - width) // 2, im_w - target_w))
    expanded_y1 = max(0, min(y1 - (target_h - height) // 2, im_h - target_h))
    return [expanded_x1, expanded_y1, expanded_x1 + target_w, expanded_y1 + target_h]


class DrawToolCheckBox(QCheckBox):
    checked = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateChanged.connect(self.on_state_changed)

    def mousePressEvent(self, event) -> None:
        if self.isChecked():
            return
        return super().mousePressEvent(event)

    def on_state_changed(self, state: int) -> None:
        if self.isChecked():
            self.checked.emit()

class ToolNameLabel(QLabel):
    def __init__(self, fix_width=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSizeF(TOOLNAME_POINT_SIZE)
        fmt = QFontMetrics(font)
        
        if fix_width is not None:
            self.setFixedWidth(fix_width)
            text_width = fmt.width(self.text())
            if text_width > fix_width * 0.95:
                font_size = TOOLNAME_POINT_SIZE * fix_width * 0.95 / text_width
                font.setPointSizeF(font_size)
        self.setFont(font)


def _inpaint_selector_view(source, parent):
    """Keep canvas controls in place while modeless settings are also visible."""
    selector = ConfigComboBox(scrollWidget=parent)
    selector.setModel(source.model())
    selector.setCurrentIndex(source.currentIndex())
    selector.setFixedSize(CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT)
    source.currentIndexChanged.connect(selector.setCurrentIndex)
    selector.activated.connect(source.setCurrentIndex)
    return selector
            

class InpaintPanel(Widget):

    thicknessChanged = Signal(int)

    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.thicknessSlider = PaintQSlider()
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSlider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),
            self.tr('Rectangle'),
            # self.tr('Triangle')
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        self.inpaint_layout = inpaint_layout = QHBoxLayout()
        inpaint_layout.addWidget(ToolNameLabel(100, self.tr('Inpainter')))
        self.inpainter_panel = inpainter_panel
        self.module_combobox = _inpaint_selector_view(inpainter_panel.module_combobox, self)
        inpaint_layout.addWidget(self.module_combobox)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(inpaint_layout)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(14)

    def on_thickness_changed(self):
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    @property
    def shape(self):
        return self.shapeCombobox.currentIndex()


class PenConfigPanel(Widget):
    thicknessChanged = Signal(int)
    colorChanged = Signal(list)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thicknessSlider = PaintQSlider()
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSlider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.alphaSlider = PaintQSlider()
        self.alphaSlider.setRange(0, 255)
        self.alphaSlider.setValue(255)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)

        self.colorPicker = ColorPickerLabel()
        self.colorPicker.colorChanged.connect(self.on_color_changed)
        
        color_label = ToolNameLabel(None, self.tr('Color'))
        alpha_label = ToolNameLabel(None, self.tr('Alpha'))
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.colorPicker)
        color_layout.addWidget(alpha_label)
        color_layout.addWidget(self.alphaSlider)
        
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),
            self.tr('Rectangle'),
            # self.tr('Triangle')
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(color_layout)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(20)

    def on_thickness_changed(self):
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    def on_alpha_changed(self):
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorPicker.setPickerColor(color)
        self.colorChanged.emit(color)

    def on_color_changed(self):
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorChanged.emit(color)

    @property
    def shape(self):
        return self.shapeCombobox.currentIndex()


class EraserConfigPanel(Widget):
    thicknessChanged = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thicknessSlider = PaintQSlider()
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSlider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),
            self.tr('Rectangle'),
            # self.tr('Triangle')
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(20)

    def on_thickness_changed(self):
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    @property
    def shape(self):
        return self.shapeCombobox.currentIndex()


class RectPanel(Widget):
    dilate_ksize_changed = Signal()
    method_changed = Signal(int)
    delete_btn_clicked = Signal()
    inpaint_btn_clicked = Signal()
    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dilate_label = ToolNameLabel(100, self.tr('Dilate'))
        self.dilate_slider = PaintQSlider()
        self.dilate_slider.setRange(0, 100)
        self.dilate_slider.valueChanged.connect(self.dilate_ksize_changed)
        self.methodComboBox = QComboBox()
        self.methodComboBox.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.methodComboBox.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        method_3_label = self.tr('method 3')
        if method_3_label == 'method 3':
            # Older compiled translations do not contain this newly added key.
            method_3_label = self.tr('method 1').replace('1', '3')
        self.methodComboBox.addItem(self.tr('method 1'), MASKSEG_METHOD_1)
        self.methodComboBox.addItem(self.tr('method 2'), MASKSEG_METHOD_2)
        self.methodComboBox.addItem(method_3_label, MASKSEG_METHOD_3)
        self.methodComboBox.addItem(self.tr('Use Existing Mask'), MASKSEG_EXISTING_MASK)
        self.methodComboBox.activated.connect(self.on_inpaint_seg_method_changed)
        self.autoChecker = QCheckBox(self.tr("Auto"))
        self.autoChecker.setToolTip(self.tr("run inpainting automatically."))
        self.autoChecker.stateChanged.connect(self.on_auto_changed)
        self.inpaint_btn = QPushButton(self.tr("Inpaint"))
        self.inpaint_btn.setToolTip(self.tr("Space"))
        self.inpaint_btn.clicked.connect(self.inpaint_btn_clicked)
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setToolTip(self.tr('Ctrl+D'))
        self.delete_btn.clicked.connect(self.delete_btn_clicked)
        self.btnlayout = QHBoxLayout()
        self.btnlayout.addWidget(self.inpaint_btn)
        self.btnlayout.addWidget(self.delete_btn)

        self.inpaint_layout = inpaint_layout = QHBoxLayout()
        inpaint_layout.addWidget(ToolNameLabel(100, self.tr('Inpainter')))
        self.inpainter_panel = inpainter_panel
        self.module_combobox = _inpaint_selector_view(inpainter_panel.module_combobox, self)
        inpaint_layout.addWidget(self.module_combobox)

        glayout = QGridLayout()
        glayout.addWidget(self.dilate_label, 0, 0)
        glayout.addWidget(self.dilate_slider, 0, 1)
        glayout.addWidget(self.autoChecker, 1, 0)
        glayout.addWidget(self.methodComboBox, 1, 1)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(inpaint_layout)
        layout.addLayout(glayout)
        layout.addLayout(self.btnlayout)
        layout.setSpacing(14)

    def on_inpaint_seg_method_changed(self):
        method_id = self.methodComboBox.currentData()
        if method_id is None:
            method_id = MASKSEG_METHOD_1
        pcfg.drawpanel.rectool_method = int(method_id)

    def on_auto_changed(self):
        if self.autoChecker.isChecked():
            self.inpaint_btn.hide()
            self.delete_btn.hide()
            pcfg.drawpanel.rectool_auto = True
        else:
            pcfg.drawpanel.rectool_auto = False
            self.inpaint_btn.show()
            self.delete_btn.show()

    def auto(self) -> bool:
        return self.autoChecker.isChecked()

    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return None
        ksize = self.dilate_slider.value()
        if ksize == 0:
            return mask
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        return cv2.dilate(mask, element)


class DrawingPanel(Widget):

    scale_tool_pos: QPointF = None

    def __init__(self, canvas: Canvas, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module_manager: ModuleManager = None
        self.canvas = canvas
        self.inpaint_stroke: StrokeImgItem = None
        self.rect_inpaint_dict: dict = None
        self.inpaint_mask_array: np.ndarray = None
        self.extracted_imask_array: np.ndarray = None
        self.paint_busy = False

        border_pen = QPen(INPAINT_BRUSH_COLOR, 3, Qt.PenStyle.DashLine)
        self.inpaint_mask_item: PixmapItem = PixmapItem(border_pen)
        self.scale_circle = QGraphicsEllipseItem()
        
        canvas.finish_painting.connect(self.on_finish_painting)
        canvas.finish_erasing.connect(self.on_finish_erasing)
        canvas.ctrl_relesed.connect(self.on_canvasctrl_released)
        canvas.begin_scale_tool.connect(self.on_begin_scale_tool)
        canvas.scale_tool.connect(self.on_scale_tool)
        canvas.end_scale_tool.connect(self.on_end_scale_tool)
        canvas.cancel_scale_tool.connect(self.on_cancel_scale_tool)
        canvas.scalefactor_changed.connect(self.on_canvas_scalefactor_changed)
        canvas.begin_create_rect.connect(self.on_begin_create_rect)
        canvas.end_create_rect.connect(self.on_end_create_rect)

        self.currentTool: DrawToolCheckBox = None
        self.handTool = DrawToolCheckBox()
        self.handTool.setObjectName("DrawHandTool")
        self.handTool.checked.connect(self.on_use_handtool)
        self.handTool.stateChanged.connect(self.on_handchecker_changed)
        self.inpaintTool = DrawToolCheckBox()
        self.inpaintTool.setObjectName("DrawInpaintTool")
        self.inpaintTool.checked.connect(self.on_use_inpainttool)
        self.inpaintConfigPanel = InpaintPanel(inpainter_panel)
        self.inpaintConfigPanel.thicknessChanged.connect(self.setInpaintToolWidth)
        self.inpaintConfigPanel.shapeChanged.connect(self.setInpaintShape)

        self.restoreTool = DrawToolCheckBox()
        self.restoreTool.setObjectName("DrawRestoreTool")
        self.restoreTool.checked.connect(self.on_use_restoretool)
        self.restoreConfigPanel = EraserConfigPanel()
        self.restoreConfigPanel.thicknessChanged.connect(self.setRestoreToolWidth)
        self.restoreConfigPanel.shapeChanged.connect(self.setRestoreShape)
        self.inpaintHardResetBtn = QPushButton(self.tr("Hard Reset Inpaint"))
        self.inpaintHardResetBtn.setToolTip(self.tr("Reset all inpainted areas on this page to the original image."))
        self.inpaintHardResetBtn.clicked.connect(self.hardResetInpaint)
        self.restoreConfigPanel.layout().addWidget(self.inpaintHardResetBtn)

        self.rectTool = DrawToolCheckBox()
        self.rectTool.setObjectName("DrawRectTool")
        self.rectTool.checked.connect(self.on_use_recttool)
        self.rectTool.stateChanged.connect(self.on_rectchecker_changed)
        self.rectPanel = RectPanel(inpainter_panel)
        self.rectPanel.inpaint_btn_clicked.connect(self.on_rect_inpaintbtn_clicked)
        self.rectPanel.delete_btn_clicked.connect(self.on_rect_deletebtn_clicked)
        self.rectPanel.dilate_ksize_changed.connect(self.on_rectool_ksize_changed)

        self.penTool = DrawToolCheckBox()
        self.penTool.setObjectName("DrawPenTool")
        self.penTool.checked.connect(self.on_use_pentool)
        self.penConfigPanel = PenConfigPanel()
        self.penConfigPanel.thicknessChanged.connect(self.setPenToolWidth)
        self.penConfigPanel.colorChanged.connect(self.setPenToolColor)
        self.penConfigPanel.shapeChanged.connect(self.setPenShape)

        self.eraserTool = DrawToolCheckBox()
        self.eraserTool.setObjectName("DrawEraserTool")
        self.eraserTool.checked.connect(self.on_use_erasertool)
        self.eraserConfigPanel = EraserConfigPanel()
        self.eraserConfigPanel.thicknessChanged.connect(self.setEraserToolWidth)
        self.eraserConfigPanel.shapeChanged.connect(self.setEraserShape)

        self._tool_display_names = {
            'hand': self._toolDisplayName('Hand'),
            'inpaint': self._toolDisplayName('Inpaint'),
            'restore': self._toolDisplayName('Restore'),
            'pen': self._toolDisplayName('Pen'),
            'eraser': self._toolDisplayName('Eraser'),
            'rect': self._toolDisplayName('Rectangle'),
        }

        toolboxlayout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        toolboxlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        toolboxlayout.addWidget(self.handTool)
        toolboxlayout.addWidget(self.inpaintTool)
        toolboxlayout.addWidget(self.restoreTool)
        toolboxlayout.addWidget(self.penTool)
        toolboxlayout.addWidget(self.eraserTool)
        toolboxlayout.addWidget(self.rectTool)

        self.canvas.painting_pen = self.pentool_pen = \
            QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.canvas.erasing_pen = self.erasing_pen = QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.erasertool_pen = QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.restoretool_pen = QPen(RESTORE_BRUSH_COLOR, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.inpaint_pen = QPen(INPAINT_BRUSH_COLOR, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        
        # self.setPenToolWidth(10)
        # self.setPenToolColor([0, 0, 0, 127])

        self.toolConfigStackwidget = QStackedWidget()
        self.toolConfigStackwidget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.toolConfigStackwidget.addWidget(self.inpaintConfigPanel)
        self.toolConfigStackwidget.addWidget(self.restoreConfigPanel)
        self.toolConfigStackwidget.addWidget(self.penConfigPanel)
        self.toolConfigStackwidget.addWidget(self.eraserConfigPanel)
        self.toolConfigStackwidget.addWidget(self.rectPanel)

        self.maskTransperancySlider = PaintQSlider()
        self.maskTransperancySlider.valueChanged.connect(self.canvas.setMaskTransparencyBySlider)
        masklayout = QHBoxLayout()
        masklayout.addWidget(ToolNameLabel(130, self.tr('Mask Opacity')))
        masklayout.addWidget(self.maskTransperancySlider)

        layout = QVBoxLayout(self)
        layout.addLayout(toolboxlayout)
        layout.addWidget(SeparatorWidget())
        layout.addWidget(self.toolConfigStackwidget)
        layout.addWidget(SeparatorWidget())
        layout.addLayout(masklayout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def setCurrentToolByName(self, tool_name: str):
        if self.paint_busy:
            return
        try:
            set_method = f'on_use_{tool_name}tool'
            set_method = getattr(self, set_method)
            set_method()
            if self.currentTool is not None:
                self.currentTool.setChecked(True)
        except:
            LOGGER.error(f'{set_method} not found in drawing panel')

    def shortcutSetCurrentToolByName(self, tool_name: str):
        if self.isVisible():
            self.setCurrentToolByName(tool_name)

    def _toolDisplayName(self, text: str) -> str:
        translated = self.tr(text)
        if translated != text:
            return translated
        if pcfg.display_lang == 'ko_KR':
            ko_map = {
                'Hand': '손 도구',
                'Inpaint': '인페인트',
                'Restore': '복원',
                'Pen': '펜',
                'Eraser': '지우개',
                'Rectangle': '사각형',
            }
            return ko_map.get(text, text)
        return text

    def setShortcutTip(self, tool_name: str, shortcut: str):
        try:
            tool_attr = f'{tool_name}Tool'
            tool: DrawToolCheckBox = getattr(self, tool_attr)
            tool_display_name = self._tool_display_names.get(tool_name, tool_name)
            if shortcut:
                tool.setToolTip(f'{tool_display_name} ({shortcut})')
            else:
                tool.setToolTip(tool_display_name)
        except AttributeError:
            LOGGER.error(f'{tool_attr} not found in drawing panel')

    def initDLModule(self, module_manager: ModuleManager):
        self.module_manager = module_manager
        module_manager.canvas_inpaint_finished.connect(self.on_inpaint_finished)
        module_manager.inpaint_thread.inpaint_failed.connect(self.on_inpaint_failed)

    def setPaintBusy(self, busy: bool):
        if self.paint_busy == busy:
            return
        self.paint_busy = busy
        self.canvas.paint_busy = busy
        for tool in (self.handTool, self.inpaintTool, self.restoreTool, self.penTool, self.eraserTool, self.rectTool):
            tool.setEnabled(not busy)
        self.inpaintHardResetBtn.setEnabled(not busy)
        self.toolConfigStackwidget.setEnabled(not busy)
        if not busy:
            self.restoreCurrentToolMode()

    def restoreCurrentToolMode(self):
        if self.currentTool == self.handTool:
            self.canvas.image_edit_mode = ImageEditMode.HandTool
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif self.currentTool == self.inpaintTool:
            self.canvas.image_edit_mode = ImageEditMode.InpaintTool
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self.isVisible():
                self.setInpaintCursor()
        elif self.currentTool == self.restoreTool:
            self.canvas.image_edit_mode = ImageEditMode.RestoreTool
            self.canvas.painting_pen = self.restoretool_pen
            self.canvas.erasing_pen = self.restoretool_pen
            self.canvas.painting_shape = self.restoreConfigPanel.shape
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self.isVisible():
                self.setRestoreCursor()
        elif self.currentTool == self.penTool:
            self.canvas.image_edit_mode = ImageEditMode.PenTool
            self.canvas.erasing_pen = self.erasing_pen
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self.isVisible():
                self.setPenCursor()
        elif self.currentTool == self.eraserTool:
            self.canvas.image_edit_mode = ImageEditMode.EraserTool
            self.canvas.painting_pen = self.erasertool_pen
            self.canvas.erasing_pen = self.erasertool_pen
            self.canvas.painting_shape = self.eraserConfigPanel.shape
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self.isVisible():
                self.setEraserCursor()
        elif self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self.isVisible():
                self.setCrossCursor()

    def cancelActiveOperation(self) -> bool:
        if self.paint_busy:
            return False
        canceled = False
        if self.scale_tool_pos is not None:
            self.on_cancel_scale_tool()
            canceled = True
        if self.inpaint_stroke is not None or self.rect_inpaint_dict is not None or \
                (self.inpaint_mask_item is not None and self.inpaint_mask_item.scene() == self.canvas):
            self.clearInpaintItems()
            canceled = True
        if self.canvas.cancelActivePaintGesture():
            canceled = True
        if canceled:
            self.restoreCurrentToolMode()
        return canceled

    def setInpaintToolWidth(self, width):
        self.inpaint_pen.setWidthF(width)
        pcfg.drawpanel.inpainter_width = width
        if self.isVisible():
            self.setInpaintCursor()

    def setInpaintShape(self, shape: int):
        pcfg.drawpanel.inpainter_shape = shape
        self.canvas.painting_shape = shape
        if self.isVisible():
            self.setInpaintCursor()

    def setRestoreToolWidth(self, width):
        self.restoretool_pen.setWidthF(width)
        pcfg.drawpanel.restoretool_width = self.restoretool_pen.widthF()
        if self.isVisible():
            self.setRestoreCursor()

    def setRestoreShape(self, shape: int):
        self.canvas.painting_shape = shape
        pcfg.drawpanel.restoretool_shape = shape
        if self.isVisible():
            self.setRestoreCursor()

    def setPenToolWidth(self, width):
        self.pentool_pen.setWidthF(width)
        self.erasing_pen.setWidthF(width)
        pcfg.drawpanel.pentool_width = self.pentool_pen.widthF()
        if self.isVisible():
            self.setPenCursor()

    def setEraserToolWidth(self, width):
        self.erasertool_pen.setWidthF(width)
        pcfg.drawpanel.erasertool_width = self.erasertool_pen.widthF()
        if self.isVisible():
            self.setEraserCursor()

    def setPenToolColor(self, color: Union[QColor, Tuple, List]):
        if not isinstance(color, QColor):
            color = QColor(*color)
        self.pentool_pen.setColor(color)
        pcfg.drawpanel.pentool_color = [color.red(), color.green(), color.blue(), color.alpha()]
        if self.isVisible():
            self.setPenCursor()
        self.penConfigPanel.colorPicker.setPickerColor(color)
        self.penConfigPanel.alphaSlider.setValue(color.alpha())

    def setPenShape(self, shape: int):
        self.canvas.painting_shape = shape
        pcfg.drawpanel.pentool_shape = shape
        if self.isVisible():
            self.setPenCursor()

    def setEraserShape(self, shape: int):
        self.canvas.painting_shape = shape
        pcfg.drawpanel.erasertool_shape = shape
        if self.isVisible():
            self.setEraserCursor()

    def on_use_handtool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.handTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.handTool
        pcfg.drawpanel.current_tool = ImageEditMode.HandTool
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.canvas.image_edit_mode = ImageEditMode.HandTool

    def on_use_inpainttool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.inpaintTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.inpaintTool
        pcfg.drawpanel.current_tool = ImageEditMode.InpaintTool
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        self.canvas.painting_pen = self.inpaint_pen
        self.canvas.erasing_pen = self.inpaint_pen
        self.canvas.painting_shape = self.inpaintConfigPanel.shape
        self.toolConfigStackwidget.setCurrentWidget(self.inpaintConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setInpaintCursor()

    def on_use_restoretool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.restoreTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.restoreTool
        pcfg.drawpanel.current_tool = ImageEditMode.RestoreTool
        self.canvas.image_edit_mode = ImageEditMode.RestoreTool
        self.canvas.painting_pen = self.restoretool_pen
        self.canvas.erasing_pen = self.restoretool_pen
        self.canvas.painting_shape = self.restoreConfigPanel.shape
        self.toolConfigStackwidget.setCurrentWidget(self.restoreConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setRestoreCursor()

    def on_use_pentool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.penTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.penTool
        pcfg.drawpanel.current_tool = ImageEditMode.PenTool
        self.canvas.painting_pen = self.pentool_pen
        self.canvas.painting_shape = self.penConfigPanel.shape
        self.canvas.erasing_pen = self.erasing_pen
        self.canvas.image_edit_mode = ImageEditMode.PenTool
        self.toolConfigStackwidget.setCurrentWidget(self.penConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setPenCursor()

    def on_use_erasertool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.eraserTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.eraserTool
        pcfg.drawpanel.current_tool = ImageEditMode.EraserTool
        self.canvas.painting_pen = self.erasertool_pen
        self.canvas.erasing_pen = self.erasertool_pen
        self.canvas.painting_shape = self.eraserConfigPanel.shape
        self.canvas.image_edit_mode = ImageEditMode.EraserTool
        self.toolConfigStackwidget.setCurrentWidget(self.eraserConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setEraserCursor()

    def on_use_recttool(self) -> None:
        if self.paint_busy:
            return
        if self.currentTool is not None and self.currentTool != self.rectTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.rectTool
        pcfg.drawpanel.current_tool = ImageEditMode.RectTool
        self.toolConfigStackwidget.setCurrentWidget(self.rectPanel)
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.canvas.image_edit_mode = ImageEditMode.RectTool
        self.setCrossCursor()

    def set_config(self, config: DrawPanelConfig):
        self.setPenToolWidth(config.pentool_width)
        self.setPenToolColor(config.pentool_color)
        self.penConfigPanel.thicknessSlider.setValue(int(config.pentool_width))
        self.penConfigPanel.shapeCombobox.setCurrentIndex(config.pentool_shape)

        self.setEraserToolWidth(getattr(config, 'erasertool_width', config.pentool_width))
        self.eraserConfigPanel.thicknessSlider.setValue(int(getattr(config, 'erasertool_width', config.pentool_width)))
        self.eraserConfigPanel.shapeCombobox.setCurrentIndex(getattr(config, 'erasertool_shape', config.pentool_shape))

        restore_width = getattr(config, 'restoretool_width', config.inpainter_width)
        restore_shape = getattr(config, 'restoretool_shape', config.inpainter_shape)
        self.setRestoreToolWidth(restore_width)
        self.restoreConfigPanel.thicknessSlider.setValue(int(restore_width))
        self.restoreConfigPanel.shapeCombobox.setCurrentIndex(restore_shape)
        
        self.setInpaintToolWidth(config.inpainter_width)
        self.inpaintConfigPanel.thicknessSlider.setValue(int(config.inpainter_width))
        self.inpaintConfigPanel.shapeCombobox.setCurrentIndex(config.inpainter_shape)
        
        self.rectPanel.dilate_slider.setValue(config.recttool_dilate_ksize)
        self.rectPanel.autoChecker.setChecked(config.rectool_auto)
        method_index = self.rectPanel.methodComboBox.findData(config.rectool_method)
        if method_index < 0:
            method_index = self.rectPanel.methodComboBox.findData(MASKSEG_METHOD_1)
        self.rectPanel.methodComboBox.setCurrentIndex(method_index)
        if config.current_tool == ImageEditMode.HandTool:
            self.handTool.setChecked(True)
        elif config.current_tool == ImageEditMode.InpaintTool:
            self.inpaintTool.setChecked(True)
        elif config.current_tool == ImageEditMode.RestoreTool:
            self.restoreTool.setChecked(True)
        elif config.current_tool == ImageEditMode.PenTool:
            self.penTool.setChecked(True)
        elif config.current_tool == ImageEditMode.EraserTool:
            self.eraserTool.setChecked(True)
        elif config.current_tool == ImageEditMode.RectTool:
            self.rectTool.setChecked(True)

    def get_pen_cursor(self, pen_color: QColor = None, pen_size = None, draw_shape=True, shape=PenShape.Circle) -> QCursor:
        cross_size = 31
        cross_len = cross_size // 4
        thickness = 3
        if pen_color is None:
            pen_color = QColor(self.pentool_pen.color())
        else:
            pen_color = QColor(pen_color)
        if pen_size is None:
            pen_size = self.pentool_pen.width()
        pen_size = max(MIN_PEN_SIZE, pen_size * self.canvas.scale_factor)
        draw_pen_size = min(pen_size, MAX_CURSOR_PIXMAP_SIZE - 2 * thickness)
        map_size = int(max(cross_size + 7, draw_pen_size + 2 * thickness))
        cursor_center = map_size // 2
        pen_radius = draw_pen_size / 2
        pen_color.setAlpha(127)
        pen = QPen(pen_color, thickness, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        pen.setDashPattern([3, 6])
        if draw_pen_size < 20:
            pen.setStyle(Qt.PenStyle.SolidLine)

        cur_pixmap = QPixmap(QSizeF(map_size, map_size).toSize())
        cur_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cur_pixmap)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if draw_shape:
            shape_rect = QRectF(cursor_center-pen_radius + thickness, 
                                cursor_center-pen_radius + thickness, 
                                max(1, draw_pen_size - 2*thickness),
                                max(1, draw_pen_size - 2*thickness))
            if shape == PenShape.Circle:
                painter.drawEllipse(shape_rect)
            elif shape == PenShape.Rectangle:
                painter.drawRect(shape_rect)
            else:
                raise NotImplementedError
            # elif shape == PenShape.Triangle:
                # painter.drawPolygon
        cross_left = (map_size  - 1 - cross_size) // 2 
        cross_right = map_size - cross_left

        pen = QPen(Qt.GlobalColor.white, 5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        cross_hline0 = QLineF(cross_left, cursor_center, cross_left+cross_len, cursor_center)
        cross_hline1 = QLineF(cross_right-cross_len, cursor_center, cross_right, cursor_center)
        cross_vline0 = QLineF(cursor_center, cross_left, cursor_center, cross_left+cross_len)
        cross_vline1 = QLineF(cursor_center, cross_right-cross_len, cursor_center, cross_right)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        pen.setWidth(3)
        pen.setColor(Qt.GlobalColor.black)
        painter.setPen(pen)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        painter.end()
        return QCursor(cur_pixmap)

    def on_incre_pensize(self):
        self.scalePen(1.1)

    def on_decre_pensize(self):
        self.scalePen(0.9)
        pass

    def scalePen(self, scale_factor):
        if self.paint_busy:
            return
        if self.currentTool == self.penTool:
            val = self.pentool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            new_val = int(np.clip(new_val, MIN_PEN_SIZE, MAX_PEN_SIZE))
            self.penConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setPenToolWidth(self.penConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.eraserTool:
            val = self.erasertool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            new_val = int(np.clip(new_val, MIN_PEN_SIZE, MAX_PEN_SIZE))
            self.eraserConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setEraserToolWidth(self.eraserConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.inpaintTool:
            val = self.inpaint_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            new_val = int(np.clip(new_val, MIN_PEN_SIZE, MAX_PEN_SIZE))
            self.inpaintConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setInpaintToolWidth(self.inpaintConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.restoreTool:
            val = self.restoretool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            new_val = int(np.clip(new_val, MIN_PEN_SIZE, MAX_PEN_SIZE))
            self.restoreConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setRestoreToolWidth(self.restoreConfigPanel.thicknessSlider.value())

    def showEvent(self, event) -> None:
        if self.currentTool is not None:
            self.currentTool.setChecked(False)
            self.currentTool.setChecked(True)
        return super().showEvent(event)

    def on_finish_painting(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        if not self.canvas.imgtrans_proj.img_valid:
            self.canvas.removeItem(stroke_item)
            return
        if self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if rect is not None:
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg))
            self.canvas.removeItem(stroke_item)
        elif self.currentTool == self.inpaintTool:
            self.inpaint_stroke = stroke_item
            if self.canvas.gv.ctrl_pressed:
                return
            else:
                self.runInpaint()
        elif self.currentTool == self.restoreTool:
            self.restoreInpaintedArea(stroke_item)

    def on_finish_erasing(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        if self.currentTool in (self.inpaintTool, self.restoreTool):
            self.restoreInpaintedArea(stroke_item)

        elif self.currentTool in (self.penTool, self.eraserTool):
            rect, _, qimg = stroke_item.clip()
            if self.canvas.erase_img_key is not None:
                self.canvas.drawingLayer.removeQImage(self.canvas.erase_img_key)
                self.canvas.erase_img_key = None
                self.canvas.stroke_img_item = None
            if rect is not None:
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg, True))
        
    def canHardResetInpaint(self) -> bool:
        project = self.canvas.imgtrans_proj
        if project is None or project.img_array is None or project.inpainted_array is None or project.mask_array is None:
            return False
        if project.img_array.shape != project.inpainted_array.shape:
            return True
        if not np.array_equal(project.img_array, project.inpainted_array):
            return True
        return bool(np.any(project.mask_array))

    def hardResetInpaint(self) -> bool:
        if self.paint_busy or not self.canHardResetInpaint():
            return False
        self.cancelActiveOperation()
        self.canvas.push_undo_command(InpaintHardResetCommand(self.canvas))
        return True

    def restoreInpaintedArea(self, stroke_item: StrokeImgItem):
        project = self.canvas.imgtrans_proj
        if project is None or project.img_array is None or project.inpainted_array is None or project.mask_array is None:
            self.canvas.removeItem(stroke_item)
            return

        rect, stroke_mask, _ = stroke_item.clip(mask_only=True)
        if stroke_mask is None:
            self.canvas.removeItem(stroke_item)
            return

        mask_h, mask_w = stroke_mask.shape[:2]
        mask_x, mask_y = rect[0], rect[1]
        inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
        y1, y2 = inpaint_rect[1], inpaint_rect[3]
        x1, x2 = inpaint_rect[0], inpaint_rect[2]

        current_mask = project.mask_array[y1:y2, x1:x2]
        restore_area = (stroke_mask > 0) & (current_mask > 0)
        if not np.any(restore_area):
            self.canvas.removeItem(stroke_item)
            return

        restored_img = np.copy(project.inpainted_array[y1:y2, x1:x2])
        restore_masked_pixels(restored_img, project.img_array[y1:y2, x1:x2], restore_area)

        restored_mask = np.copy(current_mask)
        restored_mask[restore_area] = 0
        self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, restored_img, restored_mask, inpaint_rect))
        self.canvas.removeItem(stroke_item)


    def runInpaint(self, inpaint_dict=None):
        if self.paint_busy:
            return

        if inpaint_dict is None:
            if self.inpaint_stroke is None:
                return
            elif self.inpaint_stroke.parentItem() is None:
                logger.warning("inpainting goes wrong")
                self.clearInpaintItems()
                return
                
            rect, mask, _ = self.inpaint_stroke.clip(mask_only=True)
            if mask is None:
                self.clearInpaintItems()
                return
            # we need to enlarge the mask window a bit to get better results
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            img = self.canvas.imgtrans_proj.inpainted_array
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            rect_enlarged = enlarge_window(inpaint_rect, img.shape[1], img.shape[0])
            top = mask_y - rect_enlarged[1]
            bottom = rect_enlarged[3] - inpaint_rect[3]
            left = mask_x - rect_enlarged[0]
            right = rect_enlarged[2] - inpaint_rect[2]

            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            inpaint_rect = rect_enlarged
            img = img[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_dict = {'img': img, 'mask': mask, 'inpaint_rect': inpaint_rect}

        self.canvas.image_edit_mode = ImageEditMode.NONE
        self.setPaintBusy(True)
        try:
            self.module_manager.canvas_inpaint(inpaint_dict)
        except Exception:
            self.setPaintBusy(False)
            self.clearInpaintItems()
            raise

    def on_inpaint_finished(self, inpaint_dict):
        try:
            inpainted = inpaint_dict['inpainted']
            inpaint_rect = inpaint_dict['inpaint_rect']
            mask = inpaint_dict['mask']
            feather_radius = int(inpaint_dict.get('feather_radius', 0))
            if feather_radius > 0:
                inpainted, mask = feather_inpaint_result(
                    inpaint_dict['img'],
                    inpainted,
                    mask,
                    radius=feather_radius,
                )
            mask_array = self.canvas.imgtrans_proj.mask_array
            existing_mask = mask_array[
                inpaint_rect[1]:inpaint_rect[3],
                inpaint_rect[0]:inpaint_rect[2],
            ]
            mask = cv2.bitwise_or(mask, existing_mask)
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, inpainted, mask, inpaint_rect))
        finally:
            self.setPaintBusy(False)
            self.clearInpaintItems()

    def on_inpaint_failed(self):
        self.setPaintBusy(False)
        self.clearInpaintItems()

    def on_canvasctrl_released(self):
        if self.isVisible() and self.currentTool == self.inpaintTool and not self.paint_busy:
            self.runInpaint()

    def on_begin_scale_tool(self, pos: QPointF):
        
        if self.currentTool == self.penTool:
            circle_pen = QPen(self.pentool_pen)
        elif self.currentTool == self.eraserTool:
            circle_pen = QPen(self.erasertool_pen)
        elif self.currentTool == self.inpaintTool:
            circle_pen = QPen(self.inpaint_pen)
        elif self.currentTool == self.restoreTool:
            circle_pen = QPen(self.restoretool_pen)
        else:
            return
        pen_radius = circle_pen.widthF() / 2 * self.canvas.scale_factor
        
        r, g, b, a = circle_pen.color().getRgb()

        circle_pen.setWidth(3)
        circle_pen.setStyle(Qt.PenStyle.DashLine)
        circle_pen.setDashPattern([3, 6])
        self.scale_circle.setPen(circle_pen)
        self.scale_circle.setBrush(QBrush(QColor(r, g, b, 127)))
        self.scale_circle.setPos(pos - QPointF(pen_radius, pen_radius))
        pen_size = 2 * pen_radius
        self.scale_circle.setRect(0, 0, pen_size, pen_size)
        self.scale_tool_pos = pos - QPointF(pen_size, pen_size)
        self.canvas.addItem(self.scale_circle)
        self.setCrossCursor()
        
    def setCrossCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(draw_shape=False))

    def on_scale_tool(self, pos: QPointF):
        if self.scale_tool_pos is None:
            return
        radius = pos.x() - self.scale_tool_pos.x()
        radius = max(min(radius, MAX_PEN_SIZE * self.canvas.scale_factor), MIN_PEN_SIZE * self.canvas.scale_factor)
        self.scale_circle.setRect(0, 0, radius, radius)

    def on_end_scale_tool(self):
        if self.scale_tool_pos is None:
            return
        circle_size = int(self.scale_circle.rect().width() / self.canvas.scale_factor)
        self.scale_tool_pos = None
        self.canvas.removeItem(self.scale_circle)

        if self.currentTool == self.penTool:
            self.setPenToolWidth(circle_size)
            self.penConfigPanel.thicknessSlider.setValue(circle_size)
            self.setPenCursor()
        elif self.currentTool == self.eraserTool:
            self.setEraserToolWidth(circle_size)
            self.eraserConfigPanel.thicknessSlider.setValue(circle_size)
            self.setEraserCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintToolWidth(circle_size)
            self.inpaintConfigPanel.thicknessSlider.setValue(circle_size)
            self.setInpaintCursor()
        elif self.currentTool == self.restoreTool:
            self.setRestoreToolWidth(circle_size)
            self.restoreConfigPanel.thicknessSlider.setValue(circle_size)
            self.setRestoreCursor()

    def on_cancel_scale_tool(self):
        if self.scale_tool_pos is None:
            return
        self.scale_tool_pos = None
        if self.scale_circle.scene() == self.canvas:
            self.canvas.removeItem(self.scale_circle)
        if self.currentTool == self.penTool:
            self.setPenCursor()
        elif self.currentTool == self.eraserTool:
            self.setEraserCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintCursor()
        elif self.currentTool == self.restoreTool:
            self.setRestoreCursor()

    def on_canvas_scalefactor_changed(self):
        if not self.isVisible():
            return
        if self.currentTool == self.penTool:
            self.setPenCursor()
        elif self.currentTool == self.eraserTool:
            self.setEraserCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintCursor()
        elif self.currentTool == self.restoreTool:
            self.setRestoreCursor()

    def setPenCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(shape=self.penConfigPanel.shape))

    def setEraserCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(QColor(230, 230, 230), self.erasertool_pen.width(), shape=self.eraserConfigPanel.shape))

    def setInpaintCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(INPAINT_BRUSH_COLOR, self.inpaint_pen.width(), shape=self.inpaintConfigPanel.shape))

    def setRestoreCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(RESTORE_BRUSH_COLOR, self.restoretool_pen.width(), shape=self.restoreConfigPanel.shape))

    def on_handchecker_changed(self):
        if self.handTool.isChecked():
            self.toolConfigStackwidget.hide()
        else:
            self.toolConfigStackwidget.show()

    def on_end_create_rect(self, rect: QRectF, mode: int):
        if self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.NONE
            img = self.canvas.imgtrans_proj.inpainted_array
            im_h, im_w = img.shape[:2]

            xyxy = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]
            xyxy = np.array(xyxy)
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, im_w - 1)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, im_h - 1)
            x1, y1, x2, y2 = xyxy.astype(np.int64)
            if y2 - y1 < 2 or x2 - x1 < 2:
                if self.rect_inpaint_dict is not None:
                    self.restoreRectInpaintSelection(self.captureRectInpaintSelection())
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                return
            if mode == 0:
                im = np.copy(img[y1: y2, x1: x2])
                maskseg_method = get_maskseg_method()
                inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=self.canvas.imgtrans_proj.mask_array[y1: y2, x1: x2])
                mask = self.rectPanel.post_process_mask(inpaint_mask_array)

                bground_rgb = bub_dict['bground_rgb']
                need_inpaint = bub_dict['need_inpaint']

                inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2]}
                inpaint_dict['need_inpaint'] = need_inpaint
                inpaint_dict['bground_rgb'] = bground_rgb
                inpaint_dict['ballon_mask'] = ballon_mask
                inpaint_dict['force_inpaint'] = bool(bub_dict.get('force_inpaint', False))
                inpaint_dict['context_ratio'] = float(bub_dict.get('context_ratio', 1.0))
                inpaint_dict['feather_radius'] = int(bub_dict.get('feather_radius', 0))
                user_preview_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                user_preview_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
                preview_pixmap = ndarray2pixmap(user_preview_mask)
                if self.rectPanel.auto():
                    self.inpaint_mask_item.setPixmap(preview_pixmap)
                    self.inpaint_mask_item.setParentItem(self.canvas.baseLayer)
                    self.inpaint_mask_item.setPos(x1, y1)
                    self.inpaintRect(inpaint_dict)
                else:
                    new_state = {
                        'rect_inpaint_dict': inpaint_dict,
                        'inpaint_mask_array': inpaint_mask_array,
                        'preview_pixmap': preview_pixmap,
                        'preview_pos': QPointF(x1, y1),
                    }
                    if self.rect_inpaint_dict is None:
                        self.restoreRectInpaintSelection(new_state)
                    else:
                        self.canvas.push_undo_command(RectInpaintSelectionCommand(self, new_state))
            else:   # erasing
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                erased = self.canvas.imgtrans_proj.img_array[y1: y2, x1: x2]
                self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, erased, mask, [x1, y1, x2, y2]))
                self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.setCrossCursor()

    def on_begin_create_rect(self):
        if self.currentTool != self.rectTool or self.rectPanel.auto():
            return
        if self.rect_inpaint_dict is not None and self.inpaint_mask_item.scene() == self.canvas:
            self.canvas.removeItem(self.inpaint_mask_item)

    def _expand_rect_inpaint_context(self, inpaint_dict):
        """Pad a method-3 mask into a larger crop without changing its pixels."""
        if not inpaint_dict.get('force_inpaint', False):
            return inpaint_dict

        page_img = self.canvas.imgtrans_proj.inpainted_array
        if page_img is None:
            return inpaint_dict

        selected_rect = [int(value) for value in inpaint_dict['inpaint_rect']]
        x1, y1, x2, y2 = selected_rect
        selected_h, selected_w = y2 - y1, x2 - x1
        mask = inpaint_dict.get('mask')
        if mask is None or mask.shape[:2] != (selected_h, selected_w):
            logger.warning('Unable to expand rectangle inpaint context: mask and selection sizes differ.')
            return inpaint_dict

        context_ratio = max(float(inpaint_dict.get('context_ratio', 2.5)), 1.01)
        page_h, page_w = page_img.shape[:2]
        expanded_rect = _expand_context_rect(selected_rect, page_w, page_h, context_ratio)
        ex1, ey1, ex2, ey2 = [int(value) for value in expanded_rect]
        expanded_h, expanded_w = ey2 - ey1, ex2 - ex1
        if expanded_h <= 0 or expanded_w <= 0:
            return inpaint_dict

        offset_x, offset_y = x1 - ex1, y1 - ey1
        expanded_mask = np.zeros((expanded_h, expanded_w), dtype=mask.dtype)
        expanded_mask[offset_y:offset_y + selected_h, offset_x:offset_x + selected_w] = mask

        expanded = dict(inpaint_dict)
        expanded['img'] = np.copy(page_img[ey1:ey2, ex1:ex2])
        expanded['mask'] = expanded_mask
        expanded['inpaint_rect'] = [ex1, ey1, ex2, ey2]

        ballon_mask = inpaint_dict.get('ballon_mask')
        if ballon_mask is not None and ballon_mask.shape[:2] == (selected_h, selected_w):
            expanded_ballon_mask = np.zeros((expanded_h, expanded_w), dtype=ballon_mask.dtype)
            expanded_ballon_mask[offset_y:offset_y + selected_h, offset_x:offset_x + selected_w] = ballon_mask
            expanded['ballon_mask'] = expanded_ballon_mask
        return expanded

    def inpaintRect(self, inpaint_dict):
        img = inpaint_dict['img']
        mask = inpaint_dict['mask']
        need_inpaint = inpaint_dict['need_inpaint']
        bground_rgb = inpaint_dict['bground_rgb']
        ballon_mask = inpaint_dict['ballon_mask']
        force_inpaint = bool(inpaint_dict.get('force_inpaint', False))
        if force_inpaint:
            self.runInpaint(inpaint_dict=self._expand_rect_inpaint_context(inpaint_dict))
        elif not need_inpaint and pcfg.module.check_need_inpaint:
            bg_pixel_value = [bground_rgb[ii] for ii in range(3)]
            balloon_areas = np.where(ballon_mask > 0)
            if len(img.shape) == 3 and img.shape[2] == 4:
                avg_alpha = np.mean(img[balloon_areas][..., 3])
                avg_alpha = 0 if avg_alpha < 127 else avg_alpha
                bg_pixel_value.append(avg_alpha)
            bg_pixel_value = np.array(np.round(bg_pixel_value), dtype=np.uint8)
            img[balloon_areas] = bg_pixel_value
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, img, mask, inpaint_dict['inpaint_rect'], merge_existing_mask=True))
            self.clearInpaintItems()
        else:
            self.runInpaint(inpaint_dict=inpaint_dict)

    def on_rect_inpaintbtn_clicked(self):
        if self.rect_inpaint_dict is not None:
            self.inpaintRect(self.rect_inpaint_dict)

    def on_rect_deletebtn_clicked(self):
        self.clearInpaintItems()

    def on_rectool_ksize_changed(self):
        pcfg.drawpanel.recttool_dilate_ksize = self.rectPanel.dilate_slider.value()
        if self.currentTool != self.rectTool or self.inpaint_mask_array is None or self.inpaint_mask_item is None:
            return
        mask = self.rectPanel.post_process_mask(self.inpaint_mask_array)
        self.rect_inpaint_dict['mask'] = mask
        user_preview_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        user_preview_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
        self.inpaint_mask_item.setPixmap(ndarray2pixmap(user_preview_mask))

    def on_rectchecker_changed(self):
        if not self.rectTool.isChecked():
            self.clearInpaintItems()

    @staticmethod
    def _copyRectInpaintSelection(state):
        if state is None:
            return None
        return {
            'rect_inpaint_dict': copy.deepcopy(state['rect_inpaint_dict']),
            'inpaint_mask_array': np.copy(state['inpaint_mask_array']),
            'preview_pixmap': state['preview_pixmap'].copy(),
            'preview_pos': QPointF(state['preview_pos']),
        }

    def captureRectInpaintSelection(self):
        if self.rect_inpaint_dict is None:
            return None
        state = {
            'rect_inpaint_dict': self.rect_inpaint_dict,
            'inpaint_mask_array': self.inpaint_mask_array,
            'preview_pixmap': self.inpaint_mask_item.pixmap(),
            'preview_pos': self.inpaint_mask_item.pos(),
        }
        return self._copyRectInpaintSelection(state)

    def restoreRectInpaintSelection(self, state):
        state = self._copyRectInpaintSelection(state)
        if self.inpaint_mask_item.scene() == self.canvas:
            self.canvas.removeItem(self.inpaint_mask_item)

        self.rect_inpaint_dict = None
        self.inpaint_mask_array = None
        if state is not None:
            self.rect_inpaint_dict = state['rect_inpaint_dict']
            self.inpaint_mask_array = state['inpaint_mask_array']
            self.inpaint_mask_item.setPixmap(state['preview_pixmap'])
            self.inpaint_mask_item.setParentItem(self.canvas.baseLayer)
            self.inpaint_mask_item.setPos(state['preview_pos'])

        if self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.setCrossCursor()

    def hideEvent(self, e) -> None:
        if not self.paint_busy:
            self.cancelActiveOperation()
        self.clearInpaintItems()
        return super().hideEvent(e)

    def clearInpaintItems(self):

        self.rect_inpaint_dict = None
        self.inpaint_mask_array = None
        if self.inpaint_mask_item is not None:
            if self.inpaint_mask_item.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_mask_item)
            if self.rectTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.RectTool    
            
        if self.inpaint_stroke is not None:
            if self.inpaint_stroke.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_stroke)
            self.inpaint_stroke = None
            if self.inpaintTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.InpaintTool

    def handle_page_changed(self):
        self.setPaintBusy(False)
        self.clearInpaintItems()
