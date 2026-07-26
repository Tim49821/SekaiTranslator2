import copy
import sys
from typing import List

from qtpy.QtWidgets import QLineEdit, QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame, QComboBox, QApplication, QPushButton, QLabel, QGroupBox, QCheckBox, QSlider
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QFocusEvent, QMouseEvent, QTextCursor, QKeyEvent

from utils import shared
from utils import config as C
from utils.font_loader import CustomFontOption, unique_custom_font_options
from utils.fontformat import FontFormat, px2pt, LineSpacingType
from .custom_widget import Widget, ColorPickerLabel, ClickableLabel, CheckableLabel, TextCheckerLabel, AlignmentChecker, QFontChecker, SizeComboBox, SizeControlLabel, NoBorderPushBtn
from .textitem import TextBlkItem
from .text_advanced_format import TextAdvancedFormatPanel
from .text_style_presets import TextStylePresetPanel
from . import funcmaps as FM


class LineEdit(QLineEdit):

    return_pressed_wochange = Signal()
    return_pressed = Signal()

    def __init__(self, content: str = None, parent = None):
        super().__init__(content, parent)
        self.textChanged.connect(self.on_text_changed)
        self._text_changed = False
        self.editingFinished.connect(self.on_editing_finished)
        # self.returnPressed.connect(self.on_return_pressed)

    def on_text_changed(self):
        self._text_changed = True

    def on_editing_finished(self):
        self._text_changed = False

    def focusOutEvent(self, e: QFocusEvent) -> None:
        self._text_changed = False
        return super().focusOutEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        super().keyPressEvent(e)
        if e.key() == Qt.Key.Key_Return:
            self.return_pressed.emit()
            if not self._text_changed:
                self.return_pressed_wochange.emit()


class IncrementalBtn(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(13, 13)


class FontSizeStepBtn(NoBorderPushBtn):
    def __init__(self, text: str, step: float, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.step = step
        self.setFixedSize(26, 13)


class AlignmentBtnGroup(QFrame):
    param_changed = Signal(str, int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignLeftChecker = AlignmentChecker(self)
        self.alignLeftChecker.clicked.connect(self.alignBtnPressed)
        self.alignCenterChecker = AlignmentChecker(self)
        self.alignCenterChecker.clicked.connect(self.alignBtnPressed)
        self.alignRightChecker = AlignmentChecker(self)
        self.alignRightChecker.clicked.connect(self.alignBtnPressed)
        self.alignLeftChecker.setObjectName("AlignLeftChecker")
        self.alignRightChecker.setObjectName("AlignRightChecker")
        self.alignCenterChecker.setObjectName("AlignCenterChecker")

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.alignLeftChecker)
        hlayout.addWidget(self.alignCenterChecker)
        hlayout.addWidget(self.alignRightChecker)
        hlayout.setSpacing(0)

    def alignBtnPressed(self):
        btn = self.sender()
        if btn == self.alignLeftChecker:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 0)
        elif btn == self.alignRightChecker:
            self.alignRightChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignLeftChecker.setChecked(False)
            self.param_changed.emit('alignment', 2)
        else:
            self.alignCenterChecker.setChecked(True)
            self.alignLeftChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 1)
    
    def setAlignment(self, alignment: int):
        if alignment == 0:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
        elif alignment == 1:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(True)
            self.alignRightChecker.setChecked(False)
        else:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(True)


class FormatGroupBtn(QFrame):
    param_changed = Signal(str, bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boldBtn = QFontChecker(self)
        self.boldBtn.setObjectName("FontBoldChecker")
        self.boldBtn.clicked.connect(self.setBold)
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.italicBtn.clicked.connect(self.setItalic)
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.underlineBtn.clicked.connect(self.setUnderline)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.boldBtn)
        hlayout.addWidget(self.italicBtn)
        hlayout.addWidget(self.underlineBtn)
        hlayout.setSpacing(0)

    def setBold(self):
        self.param_changed.emit('bold', self.boldBtn.isChecked())

    def setItalic(self):
        self.param_changed.emit('italic', self.italicBtn.isChecked())

    def setUnderline(self):
        self.param_changed.emit('underline', self.underlineBtn.isChecked())
    

class FontSizeBox(QFrame):
    param_changed = Signal(str, float)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.upHalfBtn = FontSizeStepBtn("+0.5", 0.5, self)
        self.upHalfBtn.setObjectName("FsizeStepHalfUp")
        self.upOneBtn = FontSizeStepBtn("+1", 1, self)
        self.upOneBtn.setObjectName("FsizeStepOneUp")
        self.downHalfBtn = FontSizeStepBtn("-0.5", -0.5, self)
        self.downHalfBtn.setObjectName("FsizeStepHalfDown")
        self.downOneBtn = FontSizeStepBtn("-1", -1, self)
        self.downOneBtn.setObjectName("FsizeStepOneDown")
        for btn in (self.upHalfBtn, self.upOneBtn, self.downHalfBtn, self.downOneBtn):
            btn.clicked.connect(self.onStepBtnClicked)
        self.upHalfBtn.setToolTip(self.tr("Increase font size by 0.5"))
        self.upOneBtn.setToolTip(self.tr("Increase font size by 1"))
        self.downHalfBtn.setToolTip(self.tr("Decrease font size by 0.5"))
        self.downOneBtn.setToolTip(self.tr("Decrease font size by 1"))
        self.fcombobox = SizeComboBox([1, 1000], 'font_size', self)
        self.fcombobox.addItems([
            "5", "5.5", "6.5", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", '22', "26", "28",
            "36", "48", "56", "72", "93", "123", "163"
        ])
        self.fcombobox.param_changed.connect(self.param_changed)

        hlayout = QHBoxLayout(self)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.upBtn)
        vlayout.addWidget(self.downBtn)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        step_layout = QVBoxLayout()
        step_up_layout = QHBoxLayout()
        step_up_layout.addWidget(self.upHalfBtn)
        step_up_layout.addWidget(self.upOneBtn)
        step_up_layout.setContentsMargins(0, 0, 0, 0)
        step_up_layout.setSpacing(0)
        step_down_layout = QHBoxLayout()
        step_down_layout.addWidget(self.downHalfBtn)
        step_down_layout.addWidget(self.downOneBtn)
        step_down_layout.setContentsMargins(0, 0, 0, 0)
        step_down_layout.setSpacing(0)
        step_layout.addLayout(step_up_layout)
        step_layout.addLayout(step_down_layout)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(0)
        step_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        size_control_layout = QHBoxLayout()
        size_control_layout.addLayout(vlayout)
        size_control_layout.addWidget(self.fcombobox)
        size_control_layout.setContentsMargins(0, 0, 0, 0)
        size_control_layout.setSpacing(0)
        hlayout.addLayout(step_layout)
        hlayout.addLayout(size_control_layout)
        hlayout.setSpacing(1)
        hlayout.setContentsMargins(0, 0, 0, 0)

    def getFontSize(self) -> str:
        return self.fcombobox.currentText()

    def currentFontSize(self):
        size = self.getFontSize()
        multi_size = False
        if "+" in size:
            size = size.strip("+")
            multi_size = True
        try:
            return float(size), multi_size
        except ValueError:
            return None, multi_size

    def formatFontSizeText(self, size: float, multi_size: bool = False) -> str:
        rounded_size = round(size, 1)
        if int(rounded_size) == rounded_size:
            text = str(int(rounded_size))
        else:
            text = f"{rounded_size:.1f}"
        if multi_size:
            text += "+"
        return text

    def applyFontSizeChange(self, newsize: float, size: float, multi_size: bool, rel_size: float = None):
        newsize = max(1, min(1000, newsize))
        if newsize == size:
            return
        if not multi_size:
            self.param_changed.emit('font_size', newsize)
            self.fcombobox.setCurrentText(self.formatFontSizeText(newsize))
        elif size > 0:
            if rel_size is None:
                rel_size = newsize / size
            self.param_changed.emit('rel_font_size', rel_size)
            self.fcombobox.setCurrentText(self.formatFontSizeText(newsize, multi_size=True))

    def onUpBtnClicked(self):
        raito = 1.25
        size, multi_size = self.currentFontSize()
        if size is None:
            return
        newsize = int(round(size * raito))
        if newsize == size:
            newsize += 1
        self.applyFontSizeChange(newsize, size, multi_size, rel_size=raito)

    def onDownBtnClicked(self):
        raito = 0.75
        size, multi_size = self.currentFontSize()
        if size is None:
            return
        newsize = int(round(size * raito))
        if newsize == size:
            newsize -= 1
        self.applyFontSizeChange(newsize, size, multi_size, rel_size=raito)

    def onStepBtnClicked(self):
        btn = self.sender()
        size, multi_size = self.currentFontSize()
        if size is None or not isinstance(btn, FontSizeStepBtn):
            return
        self.applyFontSizeChange(size + btn.step, size, multi_size)
    

class FontFamilyComboBox(QComboBox):
    param_changed = Signal(str, object)
    font_style_changed = Signal(bool, bool)
    def __init__(self, emit_if_focused=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentTextChanged.connect(self.on_fontfamily_changed)
        self.setEditable(False)
        self.emit_if_focused = emit_if_focused
        
    def apply_fontfamily(self):
        option = self.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(option, CustomFontOption) and option.family in shared.CUSTOM_FONTS:
            self.param_changed.emit('font_family', option.family)
            self.param_changed.emit('bold', option.bold)
            self.param_changed.emit('italic', option.italic)
            self.font_style_changed.emit(option.bold, option.italic)

    def update_font_list(self, font_list, preferred_font=None):
        self.currentTextChanged.disconnect(self.on_fontfamily_changed)
        current_option = self.currentData(Qt.ItemDataRole.UserRole)
        options = [
            option if isinstance(option, CustomFontOption) else CustomFontOption(str(option))
            for option in font_list
        ]
        options = unique_custom_font_options(options)
        self.clear()
        for option in options:
            self.addItem(option.display_name, option)
            if option.filename:
                self.setItemData(self.count() - 1, option.filename, Qt.ItemDataRole.ToolTipRole)
        target_index = self._find_option_index(current_option)
        if target_index < 0 and preferred_font:
            target_index = self._find_option_index(CustomFontOption(preferred_font))
        if target_index < 0 and options:
            target_index = 0
        self.setCurrentIndex(target_index)
        self.currentTextChanged.connect(self.on_fontfamily_changed)

    def _find_option_index(self, target):
        if not isinstance(target, CustomFontOption):
            return -1
        family_match = -1
        for index in range(self.count()):
            option = self.itemData(index, Qt.ItemDataRole.UserRole)
            if not isinstance(option, CustomFontOption) or option.family != target.family:
                continue
            if family_match < 0:
                family_match = index
            if option.bold == target.bold and option.italic == target.italic:
                return index
        return family_match

    def set_current_font(self, family, bold=False, italic=False):
        index = self._find_option_index(CustomFontOption(family, bold=bold, italic=italic))
        if index >= 0:
            self.setCurrentIndex(index)

    def on_fontfamily_changed(self, _family=''):
        self.apply_fontfamily()


class FontFormatPanel(Widget):
    import_font_clicked = Signal()
    open_font_folder_clicked = Signal()
    
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    global_format: FontFormat = None
    restoring_textblk: bool = False

    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.familybox = FontFamilyComboBox(emit_if_focused=True, parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_param_changed)
        self.familybox.font_style_changed.connect(self.on_font_style_changed)
        self.familybox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.importFontBtn = QPushButton(self.tr("Import Font"))
        self.importFontBtn.setToolTip(self.tr("Import Font"))
        self.importFontBtn.clicked.connect(lambda _checked=False: self.import_font_clicked.emit())

        self.openFontFolderBtn = QPushButton(self.tr("Open Fonts Folder"))
        self.openFontFolderBtn.setToolTip(self.tr("Open Fonts Folder"))
        self.openFontFolderBtn.clicked.connect(lambda _checked=False: self.open_font_folder_clicked.emit())

        self.fontsizebox = FontSizeBox(self)
        self.fontsizebox.setToolTip(self.tr("Font Size"))
        self.fontsizebox.setObjectName("FontSizeBox")
        self.fontsizebox.fcombobox.setToolTip(self.tr("Change font size"))
        self.fontsizebox.param_changed.connect(self.on_param_changed)
        
        self.lineSpacingLabel = SizeControlLabel(self, direction=1, transparent_bg=False)
        self.lineSpacingLabel.setObjectName("lineSpacingLabel")
        self.lineSpacingLabel.size_ctrl_changed.connect(self.onLineSpacingCtrlChanged)
        self.lineSpacingLabel.btn_released.connect(lambda : self.on_param_changed('line_spacing', self.lineSpacingBox.value()))

        self.lineSpacingBox = SizeComboBox([0, 100], 'line_spacing', self)
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.param_changed.connect(self.on_param_changed)
        
        self.colorPicker = ColorPickerLabel(self, param_name='frgb')
        self.colorPicker.setToolTip(self.tr("Change font color"))
        self.colorPicker.changingColor.connect(self.changingColor)
        self.colorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.colorPicker.apply_color.connect(self.on_apply_color)

        self.alignBtnGroup = AlignmentBtnGroup(self)
        self.alignBtnGroup.param_changed.connect(self.on_param_changed)

        self.formatBtnGroup = FormatGroupBtn(self)
        self.formatBtnGroup.param_changed.connect(self.on_param_changed)

        self.verticalChecker = QFontChecker(self)
        self.verticalChecker.setObjectName("FontVerticalChecker")
        self.verticalChecker.clicked.connect(lambda : self.on_param_changed('vertical', self.verticalChecker.isChecked()))

        self.strokeWidthBox = SizeComboBox([0, 10], 'stroke_width', self)
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.param_changed.connect(self.on_param_changed)

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        font = self.fontStrokeLabel.font()
        font.setPointSizeF(shared.CONFIG_FONTSIZE_CONTENT * 0.95)
        self.fontStrokeLabel.setFont(font)
        self.fontStrokeLabel.size_ctrl_changed.connect(self.strokeWidthBox.changeByDelta)
        self.fontStrokeLabel.btn_released.connect(lambda : self.on_param_changed('stroke_width', self.strokeWidthBox.value()))
        
        self.strokeColorPicker = ColorPickerLabel(self, param_name='srgb')
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.strokeColorPicker.apply_color.connect(self.on_apply_color)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.letterSpacingBox = SizeComboBox([0, 10], "letter_spacing", self)
        self.letterSpacingBox.addItems(["0.0"])
        self.letterSpacingBox.setToolTip(self.tr("Change letter spacing"))
        self.letterSpacingBox.setMinimumWidth(int(self.letterSpacingBox.height() * 2.5))
        self.letterSpacingBox.param_changed.connect(self.on_param_changed)

        self.letterSpacingLabel = SizeControlLabel(self, direction=0, transparent_bg=False)
        self.letterSpacingLabel.setObjectName("letterSpacingLabel")
        self.letterSpacingLabel.size_ctrl_changed.connect(self.letterSpacingBox.changeByDelta)
        self.letterSpacingLabel.btn_released.connect(lambda : self.on_param_changed('letter_spacing', self.letterSpacingBox.value()))

        lettersp_hlayout = QHBoxLayout()
        lettersp_hlayout.addWidget(self.letterSpacingLabel)
        lettersp_hlayout.addWidget(self.letterSpacingBox)
        lettersp_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.textstyle_panel = TextStylePresetPanel(
            self.global_fontfmt_str,
            config_name='show_text_style_preset',
            config_expand_name='expand_tstyle_panel'
        )
        self.textstyle_panel.active_text_style_label_changed.connect(self.on_active_textstyle_label_changed)
        self.textstyle_panel.active_stylename_edited.connect(self.on_active_stylename_edited)

        self.textadvancedfmt_panel = TextAdvancedFormatPanel(
            self.tr('Advanced Text Format'),
            config_name='text_advanced_format_panel',
            config_expand_name='expand_tadvanced_panel',
            on_format_changed=self.on_param_changed
        )
        color_label = self.textadvancedfmt_panel.shadow_group.color_label
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)

        color_label = self.textadvancedfmt_panel.gradient_group.start_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        color_label = self.textadvancedfmt_panel.gradient_group.end_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        self.foldTextBtn = CheckableLabel(self.tr("Unfold"), self.tr("Fold"), False)
        self.sourceBtn = TextCheckerLabel(self.tr("Source"))
        self.transBtn = TextCheckerLabel(self.tr("Translation"))

        FONTFORMAT_SPACING = 6

        vl0 = QVBoxLayout()
        vl0.addWidget(self.textstyle_panel.view_widget)
        vl0.addWidget(self.textadvancedfmt_panel.view_widget)
        vl0.setSpacing(0)
        vl0.setContentsMargins(0, 0, 0, 0)
        font_tools_layout = QHBoxLayout()
        font_tools_layout.addWidget(self.importFontBtn)
        font_tools_layout.addWidget(self.openFontFolderBtn)
        font_tools_layout.setStretch(0, 1)
        font_tools_layout.setStretch(1, 1)
        font_tools_layout.setSpacing(4)
        font_tools_layout.setContentsMargins(0, 12, 0, 0)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.addWidget(self.lineSpacingLabel)
        hl1.addWidget(self.lineSpacingBox)
        hl1.setSpacing(4)
        hl1.setContentsMargins(0, 4, 0, 0)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.setSpacing(FONTFORMAT_SPACING)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.setContentsMargins(3, 0, 3, 0)
        hl3.setSpacing(13)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setContentsMargins(0, 12, 0, 0)
        hl4.setSpacing(0)

        self.vlayout.addLayout(vl0)
        self.vlayout.addLayout(font_tools_layout)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.setContentsMargins(0, 0, 7, 0)
        self.vlayout.setSpacing(0)

        self.focusOnColorDialog = False
        C.active_format = self.global_format

    def global_mode(self):
        return id(C.active_format) == id(self.global_format)
    
    def active_text_style_label(self):
        return self.textstyle_panel.active_text_style_label

    def active_text_style_format(self):
        af = self.active_text_style_label()
        if af is not None:
            return af.fontfmt
        else:
            return None

    def on_param_changed(self, param_name: str, value):
        func = FM.handle_ffmt_change.get(param_name)
        func_kwargs = {}
        if param_name in {'font_size', 'rel_font_size'}:
            func_kwargs['clip_size'] = True
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True, **func_kwargs)
            self.update_text_style_label()
        else:
            func(param_name, value, C.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True, **func_kwargs)

    def on_font_style_changed(self, bold: bool, italic: bool):
        self.formatBtnGroup.boldBtn.setChecked(bold)
        self.formatBtnGroup.italicBtn.setChecked(italic)

    def update_text_style_label(self):
        if self.global_mode():
            active_text_style_label = self.active_text_style_label()
            if active_text_style_label is not None:
                active_text_style_label.update_style(self.global_format)

    def changingColor(self):
        self.focusOnColorDialog = True

    def onColorLabelChanged(self, is_valid=True):
        self.focusOnColorDialog = False
        if is_valid:
            sender: ColorPickerLabel = self.sender()
            rgb = sender.rgb()
            self.on_param_changed(sender.param_name, rgb)

    def on_apply_color(self, param_name, rgb):
        self.on_param_changed(param_name, rgb)

    def onLineSpacingCtrlChanged(self, delta: int):
        if C.active_format.line_spacing_type == LineSpacingType.Distance:
            mul = 0.1
        else:
            mul = 0.01
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * mul)

    def set_active_format(self, font_format: FontFormat, multi_size=False):
        C.active_format = font_format
        self.familybox.blockSignals(True)
        font_size = round(font_format.font_size, 1)
        if int(font_size) == font_size:
            font_size = str(int(font_size))
        else:
            font_size = f'{font_size:.1f}'
        if multi_size:
            font_size += "+"
        self.fontsizebox.fcombobox.setCurrentText(font_size)
        self.familybox.set_current_font(
            font_format.font_family,
            bold=font_format.bold,
            italic=font_format.italic,
        )
        self.colorPicker.setPickerColor(font_format.foreground_color())
        self.strokeColorPicker.setPickerColor(font_format.stroke_color())
        self.strokeWidthBox.setValue(font_format.stroke_width)
        self.lineSpacingBox.setValue(font_format.line_spacing)
        self.letterSpacingBox.setValue(font_format.letter_spacing)
        self.verticalChecker.setChecked(font_format.vertical)
        self.formatBtnGroup.boldBtn.setChecked(font_format.bold)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.alignBtnGroup.setAlignment(font_format.alignment)
        
        self.familybox.blockSignals(False)
        self.textadvancedfmt_panel.set_active_format(font_format)

    def set_globalfmt_title(self):
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is None:
            self.textstyle_panel.setTitle(self.global_fontfmt_str)
        else:
            title = self.global_fontfmt_str + ' - ' + active_text_style_label.fontfmt._style_name
            valid_title = self.textstyle_panel.elidedText(title)
            self.textstyle_panel.setTitle(valid_title)


    def deactivate_style_label(self):
        if self.active_text_style_label() is not None:
            self.textstyle_panel.on_stylelabel_activated(False)


    def on_active_textstyle_label_changed(self):
        '''
        merge activate textstyle into global format
        '''
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is not None:
            updated_keys = self.global_format.merge(active_text_style_label.fontfmt, compare=True)
            if self.global_mode() and len(updated_keys) > 0:
                self.set_active_format(self.global_format)
            self.set_globalfmt_title()
        else:
            if self.global_mode():
                self.set_globalfmt_title()

    def on_active_stylename_edited(self):
        if self.global_mode():
            self.set_globalfmt_title()

    def set_textblk_item(self, textblk_item: TextBlkItem = None, multi_select:bool=False):
        if textblk_item is None:
            focus_w = self.app.focusWidget()
            focus_p = None if focus_w is None else focus_w.parentWidget()
            focus_on_fmtoptions = False
            if self.focusOnColorDialog:
                focus_on_fmtoptions = True
            elif focus_p:
                if focus_p == self or focus_p.parentWidget() == self:
                    focus_on_fmtoptions = True
            if not focus_on_fmtoptions:
                # Store the current text block's format before switching to global
                if self.textblk_item is not None:
                    # Save all format properties including gradient state
                    self.textblk_item.fontformat = copy.deepcopy(C.active_format)
                self.textblk_item = None
                self.set_active_format(self.global_format, multi_select)
                self.set_globalfmt_title()
            
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                # Preserve gradient properties from the text block's format
                if hasattr(textblk_item.fontformat, 'gradient_enabled'):
                    blk_fmt.gradient_enabled = textblk_item.fontformat.gradient_enabled
                    blk_fmt.gradient_start_color = textblk_item.fontformat.gradient_start_color
                    blk_fmt.gradient_end_color = textblk_item.fontformat.gradient_end_color
                    blk_fmt.gradient_angle = textblk_item.fontformat.gradient_angle
                    blk_fmt.gradient_size = textblk_item.fontformat.gradient_size
                self.textblk_item = textblk_item
                multi_size = not textblk_item.isEditing() and textblk_item.isMultiFontSize()
                self.set_active_format(blk_fmt, multi_size)
                self.textstyle_panel.setTitle(f'TextBlock #{textblk_item.idx}')
