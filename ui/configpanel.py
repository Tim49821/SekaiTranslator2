from typing import List, Union, Tuple

from qtpy.QtWidgets import QPushButton, QKeySequenceEdit, QLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox, QScrollArea, QLineEdit
from qtpy.QtCore import Qt, Signal, QSize, QEvent, QItemSelection
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QIntValidator, QValidator, QFocusEvent

from .custom_widget import ConfigComboBox, Widget
from utils.config import pcfg
from .settings_widgets import SettingsToggle, settings_icon, settings_stylesheet, settings_row
from utils.shared import CONFIG_FONTSIZE_CONTENT, CONFIG_FONTSIZE_HEADER, CONFIG_FONTSIZE_TABLE, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN
from utils.version import APP_VERSION
from utils.network_mirrors import (
    HUGGINGFACE_MIRROR_OPTIONS,
    PYPI_MIRROR_OPTIONS,
    display_options,
    mirror_from_display,
    mirror_to_display,
)
from .module_parse_widgets import InpaintConfigPanel, TextDetectConfigPanel, TranslatorConfigPanel, OCRConfigPanel

class CustomIntValidator(QIntValidator):

    def __init__(self, bottom: int, top: int, ndigits: int = None, parent = None):
        super().__init__(bottom=bottom, top=top, parent=parent)
        self.ndigits = ndigits

    def validate(self, s: str, pos: int) -> object:
        if not s.isnumeric():
            if s != '':
                return (QValidator.State.Invalid, s, pos)
            else:
                return (QValidator.State.Intermediate, s, pos)
            
        s_ori = s
        d = int(s)
        s = str(d)
        if len(s) != len(s_ori):
            pos -= len(s_ori) - len(s)
        if len(s) > self.ndigits:
            ndel = len(s) - self.ndigits
            s = s[ndel:]
            pos -= ndel
        else:
            if d > self.top():
                if s[-1] == '0':
                    d = self.top()
                else:
                    d = d % self.top()
            d = max(d, self.bottom())
            s = str(d)
        return (QValidator.State.Acceptable, s, pos)


class PercentageLineEdit(QLineEdit):

    finish_edited = Signal(str)

    def __init__(self, default_value: str = '100', parent=None) -> None:
        super().__init__(default_value, parent=parent)
        validator = CustomIntValidator(0, 101, 3)
        self.setValidator(validator)
        self.textEdited.connect(self.on_text_edited)
        self._edited = False

    def on_text_edited(self):
        self._edited = True

    def focusOutEvent(self, e: QFocusEvent) -> None:
        if self._edited:
            text = self.text()
            if not text.isnumeric():
                text = '100'
                self.setText(text)
            self.finish_edited.emit(text)

        return super().focusOutEvent(e)


class ConfigTextLabel(QLabel):
    def __init__(self, text: str, fontsize: int, font_weight: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText(text)
        font = self.font()
        if font_weight is not None:
            font.setWeight(font_weight)
        font.setPointSizeF(fontsize)
        self.setFont(font)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setOpenExternalLinks(True)
        self.setWordWrap(True)

    def setActiveBackground(self):
        self.setStyleSheet("background-color:rgba(30, 147, 229, 51);")


class ConfigSubBlock(Widget):
    pressed = Signal(int, int)
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout=True, insert_stretch: bool = False, content_margins = (0, 4, 0, 4)) -> None:
        super().__init__()
        self.idx0: int = None
        self.idx1: int = None
        if vertical_layout:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.name = name
        if name is not None:
            textlabel = ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal)
            self.name_label = textlabel
            layout.addWidget(textlabel)
            if not vertical_layout:
                layout.setStretch(0, 1)
        if discription is not None:
            layout.addWidget(ConfigTextLabel(discription, CONFIG_FONTSIZE_CONTENT-2))
        if isinstance(widget, QWidget):
            layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignRight if name and not vertical_layout else Qt.AlignmentFlag(0))
        else:
            layout.addLayout(widget)
        self.widget = widget
        self.setContentsMargins(*content_margins)

    def setIdx(self, idx0: int, idx1: int) -> None:
        self.idx0 = idx0
        self.idx1 = idx1

    def enterEvent(self, e: QEvent) -> None:
        return super().enterEvent(e)
    

def combobox_with_label(sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True, parent: QWidget = None, insert_stretch: bool = False) -> Tuple[ConfigComboBox, QWidget]:
    combox = ConfigComboBox(fix_size=False, scrollWidget=parent)
    combox.addItems(sel)
    combox.setFixedWidth(200)
    combox.setFixedHeight(34)
    if target_block is None:
        sublock = ConfigSubBlock(combox, name, discription, vertical_layout=vertical_layout, insert_stretch=insert_stretch)
        sublock.layout().setSpacing(16)
        return combox, sublock
    else:
        layout = target_block.layout()
        layout.addSpacing(20)
        layout.addWidget(ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal))
        layout.addWidget(combox)
        return combox, target_block
    
def checkbox_with_label(name: str, discription: str = None, target_block: QWidget = None):
    checkbox = SettingsToggle()
    checkbox.setFixedSize(42, 26)
    checkbox.setAccessibleName(name)
    if target_block is None:
        target_block = ConfigSubBlock(settings_row(name, checkbox, description=discription))
    return checkbox, target_block
    


class ConfigBlock(Widget):
    sublock_pressed = Signal(int, int)

    def __init__(self, header: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.header = ConfigTextLabel(header, CONFIG_FONTSIZE_HEADER)
        self.header.setObjectName('SettingsPageHeading')
        self.vlayout = QVBoxLayout(self)
        self.vlayout.addWidget(self.header)
        self.setContentsMargins(28, 8, 28, 24)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(8)
        self.label_list = []
        self.subblock_list = []
        self.index: int = 0

    def setIndex(self, index: int):
        self.index = index

    def addLineEdit(self, name: str = None, discription: str = None, vertical_layout: bool = False):
        le = QLineEdit()
        le.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        le.setFixedHeight(45)
        sublock = ConfigSubBlock(le, name, discription, vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.addSublock(sublock)
        sublock.layout().setSpacing(20)
        return le, sublock

    def addTextLabel(self, text: str = None):
        label = ConfigTextLabel(text, CONFIG_FONTSIZE_HEADER)
        label.setObjectName('SettingsPageHeading')
        self.vlayout.addWidget(label)
        self.label_list.append(label)

    def addSublock(self, sublock: ConfigSubBlock):
        self.vlayout.addWidget(sublock)
        sublock.setIdx(self.index, len(self.label_list)-1)
        sublock.pressed.connect(lambda idx0, idx1: self.sublock_pressed.emit(idx0, idx1))
        self.subblock_list.append(sublock)

    def addCombobox(self, sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True) -> Tuple[ConfigComboBox, QWidget]:
        combox, sublock = combobox_with_label(sel, name, discription, vertical_layout, target_block, fix_size, parent=self)
        if target_block is None:
            self.addSublock(sublock)
        return combox, sublock

    def addBlockWidget(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout: bool = False) -> ConfigSubBlock:
        sublock = ConfigSubBlock(widget, name, discription, vertical_layout)
        self.addSublock(sublock)
        return sublock

    def addCheckBox(self, name: str, discription: str = None, target_block: ConfigSubBlock = None) -> QCheckBox:
        checkbox, sublock = checkbox_with_label(name, discription, target_block)
        if target_block is None:
            self.addSublock(sublock)
        return checkbox, sublock

    def getSubBlockbyIdx(self, idx: int) -> ConfigSubBlock:
        return self.subblock_list[idx]


class ConfigContent(QScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_block_list: List[ConfigBlock] = []
        self.scrollContent = Widget()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.setContentsMargins(0, 0, 0, 0)
        self.vlayout = vlayout
        self.active_label: ConfigTextLabel = None

    def addConfigBlock(self, block: ConfigBlock):
        self.vlayout.addWidget(block)
        self.config_block_list.append(block)

    def setActiveLabel(self, idx0: int, idx1: int):
        # Retain the real controls and their signal connections, but present one
        # section at a time. No scroll-to-anchor navigation or hover selection.
        for block_index, block in enumerate(self.config_block_list):
            if not hasattr(block, '_section_items'):
                section = -1
                block._section_items = []
                for i in range(block.vlayout.count()):
                    widget = block.vlayout.itemAt(i).widget()
                    if widget is None:
                        continue
                    if widget in block.label_list:
                        section = block.label_list.index(widget)
                    block._section_items.append((widget, section, not widget.isHidden()))
            overview = any(section == -1 and widget is not block.header
                           for widget, section, _ in block._section_items)
            for widget, section, enabled in block._section_items:
                selected = section == idx1 or (idx1 == -1 and not overview)
                if widget is block.header or (idx1 >= 0 and widget in block.label_list):
                    selected = False
                widget.setVisible(enabled and selected)
            block.setVisible(block_index == idx0)
        self.vlayout.activate()
        self.verticalScrollBar().setValue(0)
        self.horizontalScrollBar().setValue(0)

    def deactiveLabel(self):
        if self.active_label is not None:
            self.active_label.setStyleSheet("")
            self.active_label = None


class TableItem(QStandardItem):
    def __init__(self, text, fontsize):
        super().__init__()
        font = self.font()
        font.setPointSizeF(13)
        self.setFont(font)
        self.setText(text)
        self.setEditable(False)

    def setBold(self, bold: bool):
        font = self.font()
        font.setBold(bold)
        self.setFont(font)


class TreeModel(QStandardItemModel):
    # https://stackoverflow.com/questions/32229314/pyqt-how-can-i-set-row-heights-of-qtreeview
    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.SizeHintRole:
            size = QSize()
            item = self.itemFromIndex(index)
            size.setHeight(46 if item.parent() is None else 44)
            return size
        else:
            return super().data(index, role)


class ConfigTable(QTreeView):
    tableitem_pressed = Signal(int, int)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        treeModel = TreeModel()
        self.tm = treeModel
        self.setModel(treeModel)
        self.selected: TableItem = None
        self.last_selected: TableItem = None
        self.setHeaderHidden(True)
        self.setFixedWidth(230)
        self.setIndentation(16)
        self.setIconSize(QSize(20, 20))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def addHeader(self, header: str) -> TableItem:
        rootNode = self.model().invisibleRootItem()
        ti = TableItem(header, CONFIG_FONTSIZE_TABLE)
        rootNode.appendRow(ti)
        return ti

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        dis = deselected.indexes()
        sel = selected.indexes()
        model = self.model()
        self.last_selected = model.itemFromIndex(dis[0]) \
            if len(dis) > 0 else None
        
        self.selected = model.itemFromIndex(sel[0]) \
            if len(sel) > 0 else None
        for i in deselected.indexes():
            self.model().itemFromIndex(i).setBold(False)
        
        index = self.currentIndex()
        if index.isValid():
            self.model().itemFromIndex(index).setBold(True)
        super().selectionChanged(selected, deselected)
        if self.selected is not None:
            parent = self.selected.parent()
            self.tableitem_pressed.emit(self.selected.row() if parent is None else parent.row(),
                                        -1 if parent is None else self.selected.row())

    def setCurrentItem(self, idx0, idx1):
        item = self.tm.item(idx0, 0)
        index = (item if idx1 < 0 else item.child(idx1)).index()
        self.setCurrentIndex(index)


class ConfigPanel(Widget):

    save_config = Signal()
    unload_models = Signal()
    prepare_selected_modules = Signal()
    check_update = Signal()
    reload_textstyle = Signal(bool)
    show_only_custom_font = Signal(bool)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        settings_title = '설정' if pcfg.display_lang == 'ko_KR' else self.tr('Settings')
        self.setWindowTitle(settings_title)
        self.resize(900, 700)
        self.setMinimumSize(800, 500)
        self._visibility_toggle = None
        self._centered_once = False
        self.setObjectName("ConfigPanel")
        self.configTable = ConfigTable()
        self.configTable.tableitem_pressed.connect(self.onTableItemPressed)
        self.configContent = ConfigContent()
        self.page_heading = QLabel()
        self.page_heading.setObjectName('SettingsPageHeading')
        self.page_heading.setContentsMargins(28, 24, 28, 0)
        self.page_heading.setMinimumHeight(62)
        dlConfigPanel, dltableitem = self.addConfigBlock(self.tr('DL Module'))
        generalConfigPanel, generalTableItem = self.addConfigBlock(self.tr('General'))
        
        label_text_det = self.tr('Text Detection')
        label_text_ocr = self.tr('OCR')
        label_inpaint = self.tr('Inpaint')
        label_translator = self.tr('Translator')
        label_startup = self.tr('Startup')
        label_typesetting = self.tr('Typesetting')
        label_save = self.tr('Save')
        label_saladict = self.tr('SalaDict')
    
        dltableitem.appendRows([
            TableItem(label_text_det, CONFIG_FONTSIZE_TABLE),
            TableItem(label_text_ocr, CONFIG_FONTSIZE_TABLE),
            TableItem(label_inpaint, CONFIG_FONTSIZE_TABLE),
            TableItem(label_translator, CONFIG_FONTSIZE_TABLE),
        ])
        generalTableItem.appendRows([
            TableItem(label_startup, CONFIG_FONTSIZE_TABLE),
            TableItem(label_typesetting, CONFIG_FONTSIZE_TABLE),
            TableItem(label_save, CONFIG_FONTSIZE_TABLE),
            TableItem(label_saladict, CONFIG_FONTSIZE_TABLE),
        ])
        
        self.load_model_checker, msublock = checkbox_with_label(self.tr('Load models on demand'), discription=self.tr('Load models on demand to save memory.'))
        self.load_model_checker.stateChanged.connect(self.on_load_model_changed)
        dlConfigPanel.vlayout.addWidget(msublock)
        self.package_auto_install_checker, msublock = checkbox_with_label(
            self.tr('Auto install missing packages'),
            discription=self.tr('Install missing Python packages automatically when a selected module requires them.'),
        )
        self.package_auto_install_checker.stateChanged.connect(self.on_package_auto_install_changed)
        dlConfigPanel.vlayout.addWidget(msublock)
        self.empty_runcache_checker, msublock = checkbox_with_label(self.tr('Empty cache after RUN'), discription=self.tr('Empty cache after RUN to save memory.'))
        dlConfigPanel.vlayout.addWidget(msublock)
        self.empty_runcache_checker.stateChanged.connect(self.on_runcache_changed)
        self.prepare_modules_btn = QPushButton(parent=self)
        self.prepare_modules_btn.setMinimumHeight(34)
        self.prepare_modules_btn.setText(self.tr('Prepare Selected Modules'))
        self.prepare_modules_btn.clicked.connect(self.prepare_selected_modules)
        dlConfigPanel.vlayout.addWidget(self.prepare_modules_btn)
        self.unload_model_btn = QPushButton(parent=self)
        self.unload_model_btn.setMinimumHeight(34)
        self.unload_model_btn.setText(self.tr('Unload All Models'))
        self.unload_model_btn.clicked.connect(self.unload_models)
        dlConfigPanel.vlayout.addWidget(self.unload_model_btn)

        dlConfigPanel.addTextLabel(label_text_det)
        self.detect_config_panel = TextDetectConfigPanel(self.tr('Detector'), scrollWidget=self, compact=True)
        self.detect_sub_block = dlConfigPanel.addBlockWidget(self.detect_config_panel)
        self.detect_config_panel.keep_existing_checker.clicked.connect(self.on_keepline_clicked)

        dlConfigPanel.addTextLabel(label_text_ocr)
        self.ocr_config_panel = OCRConfigPanel(self.tr('OCR'), scrollWidget=self, compact=True)
        self.ocr_sub_block = dlConfigPanel.addBlockWidget(self.ocr_config_panel)

        dlConfigPanel.addTextLabel(label_inpaint)
        self.inpaint_config_panel = InpaintConfigPanel(self.tr('Inpainter'), scrollWidget=self, compact=True)
        self.inpaint_sub_block = dlConfigPanel.addBlockWidget(self.inpaint_config_panel)
        self.inpaint_config_panel.filter_mask_by_bboxes_checker.clicked.connect(self.on_filter_mask_by_bboxes_clicked)

        dlConfigPanel.addTextLabel(label_translator)
        self.trans_config_panel = TranslatorConfigPanel(label_translator, scrollWidget=self, compact=True)
        self.trans_sub_block = dlConfigPanel.addBlockWidget(self.trans_config_panel)

        generalConfigPanel.addTextLabel(label_startup)
        self.open_on_startup_checker, _ = generalConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)
        self.check_update_on_startup_checker, _ = generalConfigPanel.addCheckBox(self.tr('Check upstream release on startup'))
        self.check_update_on_startup_checker.stateChanged.connect(self.on_check_update_onstartup_changed)

        update_status_widget = QWidget()
        update_status_layout = QVBoxLayout(update_status_widget)
        update_status_layout.setContentsMargins(0, 0, 0, 0)
        update_status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.check_update_btn = QPushButton(parent=self)
        self.check_update_btn.setFixedWidth(160)
        self.check_update_btn.setText(self.tr('Check upstream release'))
        self.check_update_btn.clicked.connect(self.check_update)
        self.current_version_label = ConfigTextLabel(
            self.tr('Current version: ') + APP_VERSION,
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        self.latest_version_label = ConfigTextLabel(
            self.tr('Latest upstream version: ') + self.tr('Not checked'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        update_status_layout.addWidget(self.check_update_btn)
        update_status_layout.addWidget(self.current_version_label)
        update_status_layout.addWidget(self.latest_version_label)
        generalConfigPanel.addBlockWidget(update_status_widget)

        none_label = self.tr('None')
        self.huggingface_mirror_combobox, _ = generalConfigPanel.addCombobox(
            display_options(HUGGINGFACE_MIRROR_OPTIONS, none_label=none_label),
            self.tr('Huggingface Mirrors'),
            fix_size=False,
        )
        self.huggingface_mirror_combobox.setFixedWidth(200)
        self.huggingface_mirror_combobox.currentTextChanged.connect(self.on_huggingface_mirror_changed)
        self.pypi_mirror_combobox, _ = generalConfigPanel.addCombobox(
            display_options(PYPI_MIRROR_OPTIONS, none_label=none_label),
            self.tr('PyPI Mirrors'),
            fix_size=False,
        )
        self.pypi_mirror_combobox.setFixedWidth(200)
        self.pypi_mirror_combobox.currentTextChanged.connect(self.on_pypi_mirror_changed)

        generalConfigPanel.addTextLabel(label_typesetting)
        dec_program_str = self.tr('decide by program')
        use_global_str = self.tr('use global setting')

        global_fntfmt_widget = QWidget()
        global_fntfmt_layout = QGridLayout(global_fntfmt_widget)
        global_fntfmt_layout.setSpacing(0)
        global_fntfmt_layout.setContentsMargins(0, 0, 0, 0)
        global_fntfmt_widget.setContentsMargins(0, 0, 0, 0)

        b = generalConfigPanel.addBlockWidget(global_fntfmt_widget)
        b.layout().setContentsMargins(0, 0, 0, 0)
        b.setContentsMargins(0, 0, 0, 0)
        self.let_fntsize_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Size'), parent=self, insert_stretch=True)
        global_fntfmt_layout.addWidget(sublock, 0, 0)

        self.let_fntsize_combox.activated.connect(self.on_fntsize_flag_changed)
        self.let_fntstroke_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Size'), parent=self, insert_stretch=True)
        self.let_fntstroke_combox.activated.connect(self.on_fntstroke_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 0)
        
        self.let_fntcolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Color'), parent=self, insert_stretch=True)
        self.let_fntcolor_combox.activated.connect(self.on_fontcolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 0)
        self.let_fnt_scolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Color'), parent=self, insert_stretch=True)
        self.let_fnt_scolor_combox.activated.connect(self.on_font_scolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 3, 0)

        self.let_effect_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Effect'), parent=self, insert_stretch=True)
        self.let_effect_combox.activated.connect(self.on_effect_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 4, 0)
        self.let_alignment_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Alignment'), parent=self, insert_stretch=True)
        self.let_alignment_combox.activated.connect(self.on_alignment_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 5, 0)

        self.let_writing_mode_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Writing-mode'), parent=self, insert_stretch=True)
        self.let_writing_mode_combox.activated.connect(self.on_writing_mode_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 6, 0)
        self.let_family_combox, sublock = combobox_with_label([self.tr('Keep existing'), self.tr('Always use global setting')], self.tr('Font Family'), parent=self, insert_stretch=True)
        self.let_family_combox.activated.connect(self.on_family_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 7, 0)

        self.let_autolayout_checker, sublock = generalConfigPanel.addCheckBox(self.tr('Auto layout'), 
                discription=self.tr('Split translation into multi-lines according to the extracted balloon region.'))

        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)
        self.let_uppercase_checker, _ = generalConfigPanel.addCheckBox(self.tr('To uppercase'))
        self.let_uppercase_checker.stateChanged.connect(self.on_uppercase_changed)

        self.let_textstyle_indep_checker, _ = generalConfigPanel.addCheckBox(self.tr('Independent text styles for each projects'))
        self.let_textstyle_indep_checker.stateChanged.connect(self.on_textstyle_indep_changed)

        self.let_show_only_custom_fonts, sublock = generalConfigPanel.addCheckBox(self.tr("Show only custom fonts"))
        self.let_show_only_custom_fonts.stateChanged.connect(self.on_show_only_custom_fonts)
        self.let_show_only_custom_fonts.setChecked(True)
        sublock.hide()

        generalConfigPanel.addTextLabel(label_save)
        self.rst_imgformat_combobox, imsave_sublock = generalConfigPanel.addCombobox(['PNG', 'JPG', 'WEBP', 'JXL'], self.tr('Result image format'))
        self.rst_imgformat_combobox.activated.connect(self.on_rst_imgformat_changed)
        self.rst_imgquality_edit = PercentageLineEdit('100')
        self.rst_imgquality_edit.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.rst_imgquality_edit.finish_edited.connect(self.on_edit_quality_changed)

        sublock = ConfigSubBlock(self.rst_imgquality_edit, self.tr('Quality'), vertical_layout=False)
        self.rst_imgquality_edit.setFixedHeight(34)
        generalConfigPanel.addSublock(sublock)

        self.intermediate_imgformat_combobox, intermediate_imsave_sublock = generalConfigPanel.addCombobox(['PNG', 'JXL'], self.tr('Intermediate image format'))
        self.intermediate_imgformat_combobox.activated.connect(self.on_intermediate_imgformat_changed)

        generalConfigPanel.addTextLabel(label_saladict)

        sublock = ConfigSubBlock(ConfigTextLabel(self.tr("<a href=\"https://github.com/dmMaze/BallonsTranslator/tree/master/doc/saladict.md\">Installation guide</a>"), CONFIG_FONTSIZE_CONTENT - 2), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)

        self.selectext_minimenu_checker, _ = generalConfigPanel.addCheckBox(self.tr('Show mini menu when selecting text.'))
        self.selectext_minimenu_checker.stateChanged.connect(self.on_selectext_minimenu_changed)
        self.saladict_shortcut = QKeySequenceEdit("ALT+W", self)
        self.saladict_shortcut.keySequenceChanged.connect(self.on_saladict_shortcut_changed)
        self.saladict_shortcut.setFixedWidth(200)

        sublock = ConfigSubBlock(self.saladict_shortcut, self.tr("Shortcut"), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)
        self.searchurl_combobox, _ = generalConfigPanel.addCombobox(["https://www.google.com/search?q=", "https://www.bing.com/search?q=", "https://duckduckgo.com/?q=", "https://yandex.com/search/?text=", "http://www.baidu.com/s?wd=", "https://search.yahoo.com/search;?p=", "https://www.urbandictionary.com/define.php?term="], self.tr("Search Engines"), fix_size=False)
        self.searchurl_combobox.setEditable(True)
        self.searchurl_combobox.setFixedWidth(200)
        self.searchurl_combobox.currentTextChanged.connect(self.on_searchurl_changed)

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.configTable)
        page_layout = QVBoxLayout()
        page_layout.setSpacing(0)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.page_heading)
        page_layout.addWidget(self.configContent, 1)
        hlayout.addLayout(page_layout, 1)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.configTable.expandAll()
        self.refreshSettingsStyle()
        self.configTable.setCurrentItem(0, 0)

    def refreshSettingsStyle(self):
        self.setStyleSheet(settings_stylesheet(pcfg.darkmode))
        for group, root_icon, children in (
            (0, 'settings/cube.svg', ['settings/text-recognition.svg', 'settings/scan.svg', 'settings/brush.svg', 'settings/language.svg']),
            (1, 'settings/settings.svg', ['settings/player-play.svg', 'settings/letter-t.svg', 'settings/device-floppy.svg', 'settings/puzzle.svg']),
        ):
            root = self.configTable.tm.item(group, 0)
            root.setIcon(settings_icon(root_icon, pcfg.darkmode))
            for row, icon in enumerate(children):
                root.child(row).setIcon(settings_icon(icon, pcfg.darkmode))

    def on_load_model_changed(self):
        pcfg.module.load_model_on_demand = self.load_model_checker.isChecked()

    def on_package_auto_install_changed(self):
        pcfg.package_manager.auto_install_missing_packages = self.package_auto_install_checker.isChecked()

    def on_runcache_changed(self):
        pcfg.module.empty_runcache = self.empty_runcache_checker.isChecked()

    def on_keepline_clicked(self):
        pcfg.module.keep_exist_textlines = self.detect_config_panel.keep_existing_checker.isChecked()

    def on_filter_mask_by_bboxes_clicked(self):
        pcfg.module.filter_mask_by_bboxes = self.inpaint_config_panel.filter_mask_by_bboxes_checker.isChecked()

    def addConfigBlock(self, header: str) -> Tuple[ConfigBlock, TableItem]:
        cb = ConfigBlock(header, parent=self)
        self.configContent.addConfigBlock(cb)
        cb.setIndex(len(self.configContent.config_block_list)-1)
        ti = self.configTable.addHeader(header)
        return cb, ti

    def onSublockPressed(self, idx0, idx1):
        self.configTable.setCurrentItem(idx0, idx1)
        self.configContent.deactiveLabel()

    def onTableItemPressed(self, idx0, idx1):
        item = self.configTable.tm.item(idx0, 0)
        self.page_heading.setText((item if idx1 < 0 else item.child(idx1)).text())
        self.configContent.setActiveLabel(idx0, idx1)

    def on_open_onstartup_changed(self):
        pcfg.open_recent_on_startup = self.open_on_startup_checker.isChecked()

    def on_check_update_onstartup_changed(self):
        pcfg.check_update_on_startup = self.check_update_on_startup_checker.isChecked()

    def on_huggingface_mirror_changed(self):
        pcfg.mirrors.huggingface = mirror_from_display(
            self.huggingface_mirror_combobox.currentText(),
            none_label=self.tr('None'),
        )

    def on_pypi_mirror_changed(self):
        pcfg.mirrors.pypi = mirror_from_display(
            self.pypi_mirror_combobox.currentText(),
            none_label=self.tr('None'),
        )

    def on_fntsize_flag_changed(self):
        pcfg.let_fntsize_flag = self.let_fntsize_combox.currentIndex()

    def on_fntstroke_flag_changed(self):
        pcfg.let_fntstroke_flag = self.let_fntstroke_combox.currentIndex()

    def on_autolayout_changed(self):
        pcfg.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def on_uppercase_changed(self):
        pcfg.let_uppercase_flag = self.let_uppercase_checker.isChecked()

    def on_textstyle_indep_changed(self):
        pcfg.let_textstyle_indep_flag = self.let_textstyle_indep_checker.isChecked()
        self.reload_textstyle.emit(pcfg.let_textstyle_indep_flag)

    def on_rst_imgformat_changed(self):
        pcfg.imgsave_ext = '.' + self.rst_imgformat_combobox.currentText().lower()

    def on_intermediate_imgformat_changed(self):
        pcfg.intermediate_imgsave_ext = '.' + self.intermediate_imgformat_combobox.currentText().lower()

    def on_edit_quality_changed(self, value: str):
        pcfg.imgsave_quality = int(value)

    def on_selectext_minimenu_changed(self):
        pcfg.textselect_mini_menu = self.selectext_minimenu_checker.isChecked()

    def on_saladict_shortcut_changed(self):
        kstr = self.saladict_shortcut.keySequence().toString()
        if kstr:
            pcfg.saladict_shortcut = self.saladict_shortcut.keySequence().toString()

    def on_searchurl_changed(self):
        url = self.searchurl_combobox.currentText()
        pcfg.search_url = url

    def on_fontcolor_flag_changed(self):
        pcfg.let_fntcolor_flag = self.let_fntcolor_combox.currentIndex()

    def on_font_scolor_flag_changed(self):
        pcfg.let_fnt_scolor_flag = self.let_fnt_scolor_combox.currentIndex()

    def on_alignment_flag_changed(self):
        pcfg.let_alignment_flag = self.let_alignment_combox.currentIndex()

    def on_writing_mode_flag_changed(self):
        pcfg.let_writing_mode_flag = self.let_writing_mode_combox.currentIndex()

    def on_family_flag_changed(self):
        pcfg.let_family_flag = self.let_family_combox.currentIndex()

    def on_effect_flag_changed(self):
        pcfg.let_fnteffect_flag = self.let_effect_combox.currentIndex()

    def on_show_only_custom_fonts(self):
        pcfg.let_show_only_custom_fonts_flag = True
        self.show_only_custom_font.emit(True)

    def focusOnTranslator(self):
        idx0, idx1 = self.trans_sub_block.idx0, self.trans_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnInpaint(self):
        idx0, idx1 = self.inpaint_sub_block.idx0, self.inpaint_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnDetect(self):
        idx0, idx1 = self.detect_sub_block.idx0, self.detect_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnOCR(self):
        idx0, idx1 = self.ocr_sub_block.idx0, self.ocr_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def bindVisibilityToggle(self, toggle) -> None:
        self._visibility_toggle = toggle
        toggle.toggled.connect(self.setVisible)

    def showEvent(self, e) -> None:
        self.refreshSettingsStyle()
        result = super().showEvent(e)
        if not self._centered_once and self.parentWidget() is not None:
            frame = self.frameGeometry()
            frame.moveCenter(self.parentWidget().window().frameGeometry().center())
            self.move(frame.topLeft())
            self._centered_once = True
        if self._visibility_toggle is not None:
            self._visibility_toggle.setChecked(True)
        return result

    def hideEvent(self, e) -> None:
        if self._visibility_toggle is not None:
            self._visibility_toggle.setChecked(False)
        self.save_config.emit()
        return super().hideEvent(e)
        
    def setupConfig(self):
        self.blockSignals(True)

        if pcfg.open_recent_on_startup:
            self.open_on_startup_checker.setChecked(True)
        self.check_update_on_startup_checker.setChecked(pcfg.check_update_on_startup)

        self.detect_config_panel.keep_existing_checker.setChecked(pcfg.module.keep_exist_textlines)
        self.inpaint_config_panel.filter_mask_by_bboxes_checker.setChecked(pcfg.module.filter_mask_by_bboxes)
        self.let_effect_combox.setCurrentIndex(pcfg.let_fnteffect_flag)
        self.let_fntsize_combox.setCurrentIndex(pcfg.let_fntsize_flag)
        self.let_fntstroke_combox.setCurrentIndex(pcfg.let_fntstroke_flag)
        self.let_fntcolor_combox.setCurrentIndex(pcfg.let_fntcolor_flag)
        self.let_fnt_scolor_combox.setCurrentIndex(pcfg.let_fnt_scolor_flag)
        self.let_alignment_combox.setCurrentIndex(pcfg.let_alignment_flag)
        self.let_family_combox.setCurrentIndex(pcfg.let_family_flag)
        self.let_writing_mode_combox.setCurrentIndex(pcfg.let_writing_mode_flag)
        self.let_autolayout_checker.setChecked(pcfg.let_autolayout_flag)
        self.selectext_minimenu_checker.setChecked(pcfg.textselect_mini_menu)
        self.let_uppercase_checker.setChecked(pcfg.let_uppercase_flag)
        self.let_textstyle_indep_checker.setChecked(pcfg.let_textstyle_indep_flag)
        self.saladict_shortcut.setKeySequence(pcfg.saladict_shortcut)
        self.searchurl_combobox.setCurrentText(pcfg.search_url)
        self.ocr_config_panel.restoreEmptyOCRChecker.setChecked(pcfg.restore_ocr_empty)
        self.rst_imgformat_combobox.setCurrentText(pcfg.imgsave_ext.replace('.', '').upper())
        self.intermediate_imgformat_combobox.setCurrentText(pcfg.intermediate_imgsave_ext.replace('.', '').upper())
        self.rst_imgquality_edit.setText(str(pcfg.imgsave_quality))
        self.load_model_checker.setChecked(pcfg.module.load_model_on_demand)
        self.package_auto_install_checker.setChecked(pcfg.package_manager.auto_install_missing_packages)
        self.empty_runcache_checker.setChecked(pcfg.module.empty_runcache)
        self.let_show_only_custom_fonts.setChecked(True)
        none_label = self.tr('None')
        self.huggingface_mirror_combobox.setCurrentText(
            mirror_to_display(pcfg.mirrors.huggingface, none_label=none_label)
        )
        self.pypi_mirror_combobox.setCurrentText(
            mirror_to_display(pcfg.mirrors.pypi, none_label=none_label)
        )

        self.blockSignals(False)
