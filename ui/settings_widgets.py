"""Compact settings presentation, independent of model parameter storage."""

from pathlib import Path

from qtpy.QtCore import Qt, QSize, QRectF
from qtpy.QtGui import QColor, QIcon, QPainter
from qtpy.QtWidgets import QCheckBox, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy

from utils.config import pcfg


def settings_text(english, korean):
    return korean if pcfg.display_lang == 'ko_KR' else english


class SettingsToggle(QCheckBox):
    """A real checkbox with a neutral switch appearance and keyboard support."""

    def sizeHint(self):
        return QSize(42, 26)

    def minimumSizeHint(self):
        return self.sizeHint()

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else 0.45)
        painter.setPen(Qt.PenStyle.NoPen)
        # Accent is intentionally reserved for navigation, not the form.
        track = '#64748b' if self.isChecked() else '#d5d9df'
        painter.setBrush(QColor(track))
        painter.drawRoundedRect(QRectF(1, 2, 40, 22), 11, 11)
        painter.setBrush(QColor('#ffffff'))
        painter.drawEllipse(QRectF(23 if self.isChecked() else 3, 4, 18, 18))
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor('#64748b'))
            painter.drawRoundedRect(QRectF(0, 0, 41, 25), 12, 12)


def settings_row(label, control, *, stacked=False, description=None):
    row = QWidget()
    row.setObjectName('SettingsRow')
    layout = QVBoxLayout(row) if stacked or description else QHBoxLayout(row)
    layout.setContentsMargins(0, 8, 0, 8)
    layout.setSpacing(8 if stacked else 16)
    text = QLabel(label)
    text.setWordWrap(True)
    text.setMinimumWidth(0)
    text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    text.setBuddy(control)
    control.setAccessibleName(label)
    if description:
        top = QHBoxLayout()
        top.addWidget(text, 1)
        top.addWidget(control, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(top)
        detail = QLabel(description)
        detail.setObjectName('SettingsDescription')
        detail.setWordWrap(True)
        layout.addWidget(detail)
    elif stacked:
        layout.addWidget(text)
        layout.addWidget(control)
    else:
        layout.addWidget(text, 1)
        layout.addWidget(control, 0, Qt.AlignmentFlag.AlignRight)
    return row


def settings_heading(english, korean):
    heading = QLabel(settings_text(english, korean))
    heading.setObjectName('SettingsSectionHeading')
    heading.setMinimumHeight(50)
    return heading


def settings_icon(filename, dark=False):
    """Render bundled Tabler outline icons with one consistent foreground."""
    source = QIcon(str(Path(__file__).resolve().parent.parent / 'icons' / filename))
    pixmap = source.pixmap(40, 40)
    if pixmap.isNull():
        return source
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor('#e2e8f0' if dark else '#263244'))
    painter.end()
    return QIcon(pixmap)


def settings_stylesheet(dark=False):
    surface, sidebar, text, border, selected = (
        ('#20252d', '#272d36', '#e2e8f0', '#47515f', '#254558') if dark else
        ('#ffffff', '#f6f7f9', '#17212f', '#d5dbe3', '#e0f2fe')
    )
    return f'''
    #ConfigPanel {{ background: {surface}; }}
    #ConfigPanel QWidget {{ color: {text}; font-family: "Apple SD Gothic Neo", "Noto Sans", sans-serif;
        font-size: 16px; background: transparent; }}
    #ConfigPanel ConfigContent, #ConfigPanel ConfigBlock, #ConfigPanel ConfigSubBlock,
    #ConfigPanel ConfigSubBlock:hover, #ConfigPanel #SettingsRow {{ background: {surface}; border: none; }}
    #ConfigPanel QLabel {{ background: transparent; border: none; }}
    #ConfigPanel ConfigTable {{ background: {sidebar}; border: none; border-right: 1px solid {border};
        padding: 16px 10px; outline: 0; }}
    #ConfigPanel ConfigTable::item {{ border: none; border-left: 3px solid transparent;
        padding: 5px 8px; margin: 2px 0; border-radius: 7px; color: {text}; }}
    #ConfigPanel ConfigTable::item:hover {{ background: {sidebar}; }}
    #ConfigPanel ConfigTable::item:selected {{ background: {selected}; border-left: 3px solid #38bdf8; color: {text}; }}
    #ConfigPanel ConfigTable::branch {{ background: {sidebar}; }}
    #ConfigPanel ConfigTable::branch:has-children:closed {{ image: url(icons/chevron-right.svg); }}
    #ConfigPanel ConfigTable::branch:has-children:open {{ image: url(icons/chevron-down.svg); }}
    #ConfigPanel QLabel#SettingsPageHeading {{ font-size: 22px; font-weight: 600; padding-bottom: 12px; }}
    #ConfigPanel QLabel#SettingsSectionHeading {{ font-size: 17px; font-weight: 600;
        border-top: 1px solid {border}; padding-top: 13px; margin-top: 8px; }}
    #ConfigPanel QLabel#SettingsSectionHeading[firstSection="true"] {{ border-top: none; }}
    #ConfigPanel QLabel#SettingsDescription {{ font-size: 12px; color: {'#aeb8c5' if dark else '#657183'}; }}
    #ConfigPanel QComboBox, #ConfigPanel QLineEdit, #ConfigPanel QPlainTextEdit,
    #ConfigPanel QKeySequenceEdit {{ background: {surface}; border: 1px solid {border}; border-radius: 5px;
        padding: 4px 10px; selection-background-color: #64748b; selection-color: white; }}
    #ConfigPanel QComboBox:hover, #ConfigPanel QLineEdit:hover {{ background: {surface}; border-color: {border}; }}
    #ConfigPanel QComboBox:focus, #ConfigPanel QLineEdit:focus, #ConfigPanel QPlainTextEdit:focus {{ border-color: #64748b; }}
    #ConfigPanel QComboBox::drop-down {{ border: none; width: 26px; background: transparent; }}
    #ConfigPanel QComboBox QAbstractItemView {{ background: {surface}; color: {text};
        selection-background-color: {sidebar}; selection-color: {text}; border: 1px solid {border}; }}
    #ConfigPanel QPushButton {{ background: {sidebar}; border: 1px solid {border}; border-radius: 5px;
        padding: 7px 12px; min-height: 18px; }}
    #ConfigPanel QPushButton:hover {{ border-color: #94a3b8; }}
    #ConfigPanel QScrollBar:vertical {{ width: 8px; background: {surface}; margin: 0; }}
    #ConfigPanel QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 28px; }}
    #ConfigPanel QScrollBar::add-line:vertical, #ConfigPanel QScrollBar::sub-line:vertical {{ height: 0; }}
    #ConfigPanel QScrollBar::add-page:vertical, #ConfigPanel QScrollBar::sub-page:vertical {{ background: none; }}
    '''
