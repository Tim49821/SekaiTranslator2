from typing import List, Callable

from modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR, \
    BaseTranslator, DEFAULT_DEVICE, GPUINTENSIVE_SET
from utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, NoBorderPushBtn, ParamNameLabel
from utils.shared import CONFIG_COMBOBOX_LONG, size2width, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.config import pcfg

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QLineEdit, QGridLayout, QPushButton, QInputDialog, QSizePolicy
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
from .settings_widgets import SettingsToggle, settings_row, settings_heading, settings_text


class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None, compact=False) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QVBoxLayout(self) if compact else QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=k, parent=self)
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key

        if param_key == 'chat sample':
            self.setFixedWidth(int(CONFIG_COMBOBOX_LONG * 1.2))
            self.setFixedHeight(200)
        else:
            self.setFixedWidth(CONFIG_COMBOBOX_LONG)
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(name_label)
        hlayout.addWidget(self.checker)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return param_dict['display_name']
    return param_key


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, param_dict: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(get_param_display_name(param_key, param_dict))
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamPresetManager(QWidget):
    paramwidget_edited = Signal(str, object)

    def __init__(self, param_key: str, param_dict: dict, all_params: dict, scrollWidget: QWidget = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.param_dict = param_dict
        self.all_params = all_params
        self.target_param = param_dict.get('target_param', 'style guide')
        self.protected_name = param_dict.get('protected_name', 'Default')
        self._syncing = False

        value = param_dict.get('value')
        value = value if isinstance(value, dict) else {}
        default_presets = param_dict.get('default_presets')
        self.presets = dict(default_presets) if isinstance(default_presets, dict) else {}
        saved_presets = value.get('styles')
        if isinstance(saved_presets, dict):
            self.presets.update(saved_presets)
        if self.protected_name not in self.presets:
            self.presets[self.protected_name] = ''

        self.selected = value.get('selected') or self.protected_name
        if self.selected not in self.presets:
            self.selected = self.protected_name

        active_text = self._current_target_text()
        if active_text:
            if self.selected == self.protected_name:
                self.presets[self.protected_name] = active_text
            elif active_text != self.presets.get(self.selected):
                self.presets[self.selected] = active_text

        # Compatibility for code that consumes the original style-guide manager.
        self.styles = self.presets

        self.selector = ParamComboBox(param_key, list(self.presets.keys()), size=CONFIG_COMBOBOX_LONG, scrollWidget=scrollWidget)
        self.selector.setCurrentText(self.selected)
        self.editor = ParamEditor(self.target_param)
        self.editor.setText(self.presets.get(self.selected, active_text))

        self.add_btn = QPushButton(self.tr(param_dict.get('add_button', 'Add style')))
        self.replace_btn = QPushButton(self.tr(param_dict.get('replace_button', 'Replace style')))
        self.delete_btn = QPushButton(self.tr(param_dict.get('delete_button', 'Delete style')))

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.replace_btn)
        btn_layout.addWidget(self.delete_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.selector)
        layout.addWidget(self.editor)
        layout.addLayout(btn_layout)

        self.selector.currentTextChanged.connect(self.on_selected_preset_changed)
        self.editor.textChanged.connect(self.on_editor_changed)
        self.add_btn.clicked.connect(self.on_add_preset)
        self.replace_btn.clicked.connect(self.replace_selected_preset)
        self.delete_btn.clicked.connect(self.delete_selected_preset)
        self._emit_state()
        self._emit_active_preset()

    def _current_target_text(self) -> str:
        target = self.all_params.get(self.target_param, '')
        if isinstance(target, dict):
            return str(target.get('value', ''))
        return str(target)

    def _set_editor_text(self, text: str):
        self._syncing = True
        self.editor.setText(text)
        self._syncing = False

    def _set_selector_items(self):
        self._syncing = True
        current = self.selected
        self.selector.clear()
        self.selector.addItems(list(self.presets.keys()))
        self.selector.setCurrentText(current)
        self._syncing = False

    def _state(self) -> dict:
        return {
            'selected': self.selected,
            'styles': dict(self.presets),
        }

    def _emit_state(self):
        state = self._state()
        manager_param = self.all_params.get(self.param_key)
        if isinstance(manager_param, dict):
            manager_param['value'] = state
        else:
            self.all_params[self.param_key] = state
        self.paramwidget_edited.emit(self.param_key, state)

    def _emit_active_preset(self):
        preset_text = self.editor.text()
        target = self.all_params.get(self.target_param)
        if isinstance(target, dict):
            target['value'] = preset_text
        else:
            self.all_params[self.target_param] = preset_text
        self.paramwidget_edited.emit(self.target_param, preset_text)

    def on_selected_preset_changed(self, preset_name: str):
        if self._syncing or preset_name not in self.presets:
            return
        self.selected = preset_name
        self._set_editor_text(self.presets[preset_name])
        self._emit_state()
        self._emit_active_preset()

    def on_editor_changed(self):
        if self._syncing:
            return
        self.presets[self.selected] = self.editor.text()
        self._emit_state()
        self._emit_active_preset()

    def add_preset(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return False
        self.selected = name
        self.presets[name] = self.editor.text()
        self._set_selector_items()
        self._emit_state()
        self._emit_active_preset()
        return True

    def replace_selected_preset(self):
        self.presets[self.selected] = self.editor.text()
        self._emit_state()
        self._emit_active_preset()

    def delete_selected_preset(self) -> bool:
        if self.selected == self.protected_name or len(self.presets) <= 1:
            return False
        self.presets.pop(self.selected, None)
        self.selected = self.protected_name if self.protected_name in self.presets else next(iter(self.presets))
        self._set_selector_items()
        self._set_editor_text(self.presets[self.selected])
        self._emit_state()
        self._emit_active_preset()
        return True

    def on_add_preset(self):
        title = self.param_dict.get('add_title', 'Add style guide')
        label = self.param_dict.get('name_label', 'Style name:')
        name, ok = QInputDialog.getText(self, self.tr(title), self.tr(label))
        if ok:
            self.add_preset(name)


class ParamStyleGuideManager(ParamPresetManager):
    def __init__(self, param_key: str, param_dict: dict, all_params: dict, scrollWidget: QWidget = None, *args, **kwargs):
        style_param_dict = dict(param_dict)
        style_param_dict.setdefault('target_param', 'style guide')
        style_param_dict.setdefault('add_title', 'Add style guide')
        style_param_dict.setdefault('name_label', 'Style name:')
        style_param_dict.setdefault('add_button', 'Add style')
        style_param_dict.setdefault('replace_button', 'Replace style')
        style_param_dict.setdefault('delete_button', 'Delete style')
        super().__init__(param_key, style_param_dict, all_params, scrollWidget=scrollWidget, *args, **kwargs)

    def on_selected_style_changed(self, style_name: str):
        self.on_selected_preset_changed(style_name)

    def on_add_style(self):
        self.on_add_preset()

    def on_replace_style(self):
        self.replace_selected_preset()

    def on_delete_style(self):
        self.delete_selected_preset()


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, scrollWidget: QWidget = None, *args, compact=False, detector_settings=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self) if compact else QHBoxLayout(self)
        if compact:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        self.param_layout = param_layout = QGridLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        if not compact:
            layout.addLayout(param_layout)
            layout.addStretch(-1)
        text_section_added = False

        if 'description' in params:
            self.setToolTip(params['description'])

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            if isinstance(params[param_key], dict) and params[param_key].get('hidden', False):
                continue
            display_param_name = param_key

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = get_param_display_name(param_key, param_dict)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    param_widget = ParamComboBox(
                        param_key, param_dict['options'], size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    if param_key == 'device' and DEFAULT_DEVICE == 'cpu':
                        param_dict['value'] = 'cpu'
                        for device_index, device in enumerate(param_dict['options']):
                            if device in GPUINTENSIVE_SET:
                                model = param_widget.model()
                                item = model.item(device_index, 0)
                                item.setEnabled(False)
                    param_widget.setCurrentText(str(value))
                    param_widget.setEditable(param_dict.get('editable', False))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, param_dict)
                    require_label = False

                elif param_type == 'style_guide_manager':
                    param_widget = ParamStyleGuideManager(param_key, param_dict, params, scrollWidget=scrollWidget)

                elif param_type == 'preset_manager':
                    param_widget = ParamPresetManager(param_key, param_dict, params, scrollWidget=scrollWidget)

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value, compact=compact)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if 'description' in param_dict:
                        param_widget.setToolTip(param_dict['description'])

            widget_idx = 0
            if compact and param_widget is not None:
                if detector_settings and not text_section_added and param_key in ('font size multiplier', 'font size max', 'font size min', 'mask dilate size'):
                    layout.addWidget(settings_heading('Text and mask', '텍스트와 마스크'))
                    text_section_added = True
                aliases = {
                    'detect_size': '검출 크기', 'det_rearrange_max_batches': '최대 배치 수',
                    'device': '실행 장치', 'font size multiplier': '글꼴 크기 배율',
                    'font size max': '최대 글꼴 크기', 'font size min': '최소 글꼴 크기',
                    'mask dilate size': '마스크 확장',
                }
                if param_key in aliases and display_param_name == param_key:
                    display_param_name = settings_text(param_key, aliases[param_key])
                stacked = isinstance(param_widget, (ParamEditor, ParamPresetManager, ParamCheckGroup, ParamPushButton))
                if isinstance(param_widget, (ParamComboBox, ParamLineEditor)):
                    stacked = stacked or param_widget.width() > 332
                    param_widget.setFixedHeight(34)
                if stacked:
                    param_widget.setMinimumWidth(0)
                    param_widget.setMaximumWidth(16777215)
                    param_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                    if isinstance(param_widget, ParamPresetManager):
                        for child in (param_widget.selector, param_widget.editor):
                            child.setMinimumWidth(0)
                            child.setMaximumWidth(16777215)
                            child.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                else:
                    param_widget.setFixedWidth(200)
                control = param_widget
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    control = QWidget()
                    buttons = QHBoxLayout(control)
                    buttons.setContentsMargins(0, 0, 0, 0)
                    buttons.addWidget(param_widget, 1)
                    if hasattr(param_widget, 'flush_btn'):
                        buttons.addWidget(param_widget.flush_btn)
                        param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                    if hasattr(param_widget, 'path_select_btn'):
                        buttons.addWidget(param_widget.path_select_btn)
                        param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                    stacked = True
                row = settings_row(display_param_name, control, stacked=stacked) if require_label else control
                layout.addWidget(row)
                continue
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_layout.addWidget(param_label, ii, 0)
                widget_idx = 1
            if param_widget is not None:
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(param_widget, ii, widget_idx)
                else:
                    param_layout.addLayout(pw_lo, ii, widget_idx)
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

class ModuleParseWidgets(QWidget):
    def addModulesParamWidgets(self, ocr_instance):
        self.params = ocr_instance.get_params()
        self.on_module_changed()

    def on_module_changed(self):
        self.updateModuleParamWidget()

    def updateModuleParamWidget(self):
        widget = ParamWidget(self.params, scrollWidget=self)
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, *args, compact=False, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.compact = compact
        self.detector_settings = False
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget, fix_size=not compact)
        if compact:
            self.module_combobox.setFixedWidth(200)
            self.module_combobox.setFixedHeight(34)
        self.params_layout = QHBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        if not compact:
            p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        self.module_label.setWordWrap(True)
        p_layout.addWidget(self.module_label, 1 if compact else 0)
        p_layout.addWidget(self.module_combobox)
        if not compact:
            p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(12 if compact else 30)
        if compact:
            layout.setContentsMargins(0, 0, 0, 0)
            p_layout.setContentsMargins(0, 12, 0, 12)
        self.vlayout = layout

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def addModulesParamWidgets(self, module_dict: dict):
        invalid_module_keys = []
        valid_modulekeys = self.get_valid_module_keys()

        num_widgets_before = len(self.param_widget_map)

        for module in module_dict:
            if module not in valid_modulekeys:
                invalid_module_keys.append(module)
                continue

            if module in self.param_widget_map:
                LOGGER.warning(f'duplicated module key: {module}')
                continue

            self.module_combobox.addItem(module)
            params = module_dict[module]
            if params is not None:
                self.param_widget_map[module] = None

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

        self.module_dict = module_dict

        num_widgets_after = len(self.param_widget_map)
        if num_widgets_before == 0 and num_widgets_after > 0:
            self.on_module_changed()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
        self.blockSignals(True)
        self.module_combobox.setCurrentText(module)
        self.updateModuleParamWidget()
        self.blockSignals(False)

    def updateModuleParamWidget(self):
        module = self.module_combobox.currentText()
        if self.visibleWidget is not None:
            self.visibleWidget.hide()
        if module in self.param_widget_map:
            widget: QWidget = self.param_widget_map[module]
            if widget is None:
                # lazy load widgets
                params = self.module_dict[module]
                widget = ParamWidget(params, scrollWidget=self, compact=self.compact, detector_settings=self.detector_settings)
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_pre_MT_keyword_window = Signal()
    show_MT_keyword_window = Signal()
    show_OCR_keyword_window = Signal()

    def __init__(self, module_name, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget=scrollWidget, *args, **kwargs)
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox(scrollWidget=scrollWidget, fix_size=not self.compact)
        self.target_combobox = ConfigComboBox(scrollWidget=scrollWidget, fix_size=not self.compact)
        self.replacePreMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeywordBtn.clicked.connect(self.show_pre_MT_keyword_window)
        self.replacePreMTkeywordBtn.setFixedWidth(500)
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(500)
        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(500)
        self.translateByTextblockBox = ParamCheckerBox(self.tr('Translate each text block individually'))

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target')))
        st_layout.addWidget(self.target_combobox)
        
        self.vlayout.insertLayout(1, st_layout) 
        self.vlayout.addWidget(self.translateByTextblockBox)
        self.vlayout.addWidget(self.replaceOCRkeywordBtn)
        self.vlayout.addWidget(self.replacePreMTkeywordBtn)
        self.vlayout.addWidget(self.replaceMTkeywordBtn)
        if self.compact:
            # Language controls and long actions must not form a wide toolbar.
            while st_layout.count():
                item = st_layout.takeAt(0)
                if item.widget() not in (self.source_combobox, self.target_combobox):
                    item.widget().deleteLater()
            self.vlayout.removeItem(st_layout)
            for index, label, control in (
                (1, self.tr('Source'), self.source_combobox),
                (2, self.tr('Target'), self.target_combobox),
            ):
                control.setFixedSize(200, 34)
                self.vlayout.insertWidget(index, settings_row(label, control))
            for button in (self.replaceOCRkeywordBtn, self.replacePreMTkeywordBtn, self.replaceMTkeywordBtn):
                button.setMinimumWidth(0)
                button.setMaximumWidth(16777215)
                button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                button.setToolTip(button.text())
            self._compact_checker_box(self.translateByTextblockBox)

    def _compact_checker_box(self, box):
        label = box.findChild(ParamNameLabel)
        label.setWordWrap(True)
        box.layout().setContentsMargins(0, 0, 0, 0)
        box.layout().setAlignment(Qt.AlignmentFlag(0))
        box.layout().setStretch(0, 1)

    def finishSetTranslator(self, translator: BaseTranslator):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.module_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        self.source_combobox.addItems(translator.supported_src_list)
        self.target_combobox.addItems(translator.supported_tgt_list)
        self.module_combobox.setCurrentText(translator.name)
        self.source_combobox.setCurrentText(translator.lang_source)
        self.target_combobox.setCurrentText(translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.module_combobox.blockSignals(False)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget = scrollWidget, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.filter_mask_by_bboxes_checker = QCheckBox(text=self.tr('Filter mask by text boxes'))
        self.vlayout.addWidget(self.needInpaintChecker)
        self.vlayout.addWidget(self.filter_mask_by_bboxes_checker)
        if self.compact:
            label = self.needInpaintChecker.findChild(ParamNameLabel)
            label.setWordWrap(True)
            self.needInpaintChecker.layout().setContentsMargins(0, 0, 0, 0)
            self.needInpaintChecker.layout().setAlignment(Qt.AlignmentFlag(0))
            self.needInpaintChecker.layout().setStretch(0, 1)
            self.vlayout.removeWidget(self.filter_mask_by_bboxes_checker)
            text = self.filter_mask_by_bboxes_checker.text()
            self.filter_mask_by_bboxes_checker.setText('')
            self.vlayout.addWidget(settings_row(text, self.filter_mask_by_bboxes_checker))

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, scrollWidget = scrollWidget, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        if self.compact:
            self.detector_settings = True
            self.module_label.setText(settings_text('Detection model', '검출 모델'))
            self.keep_existing_checker = SettingsToggle()
            self.keep_existing_checker.setFixedSize(42, 26)
            heading = settings_heading('Detection', '검출')
            heading.setProperty('firstSection', True)
            self.vlayout.insertWidget(1, heading)
            self.vlayout.insertWidget(2, settings_row(settings_text('Keep Existing Lines', '기존 텍스트 줄 유지'), self.keep_existing_checker))
            self.vlayout.setSpacing(4)
        else:
            self.keep_existing_checker = QCheckBox(text=self.tr('Keep Existing Lines'))
            self.p_layout.insertWidget(2, self.keep_existing_checker)
        

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule
        self.restoreEmptyOCRChecker = QCheckBox(self.tr("Delete and restore region where OCR return empty string."), self)
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)
        # 字体检测选项
        self.fontDetectChecker = QCheckBox(self.tr("Font Detection"), self)
        self.fontDetectChecker.setChecked(pcfg.module.ocr_font_detect)
        self.fontDetectChecker.clicked.connect(self.on_fontdetect_changed)
        self.vlayout.addWidget(self.fontDetectChecker)
        if self.compact:
            for checkbox in (self.restoreEmptyOCRChecker, self.fontDetectChecker):
                self.vlayout.removeWidget(checkbox)
                label = checkbox.text()
                checkbox.setText('')
                self.vlayout.addWidget(settings_row(label, checkbox))

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()

    def on_fontdetect_changed(self):
        pcfg.module.ocr_font_detect = self.fontDetectChecker.isChecked()
