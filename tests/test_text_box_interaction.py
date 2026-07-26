import os
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from qtpy.QtCore import QPointF
from qtpy.QtGui import QTransform
from qtpy.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene

from ui.textitem import TextBlkItem
from utils.textblock import TextBlock


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_text_block(rect):
    x, y, width, height = rect
    block = TextBlock([x, y, x + width, y + height])
    block.set_lines_by_xywh(rect)
    block._bounding_rect = list(rect)
    return block


class TextBoxInteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = ensure_app()

    def test_nested_smaller_box_remains_clickable_when_larger_box_was_added_later(self):
        scene = QGraphicsScene()
        text_layer = QGraphicsPixmapItem()
        scene.addItem(text_layer)

        smaller = TextBlkItem(make_text_block([30, 30, 20, 20]), idx=0)
        larger = TextBlkItem(make_text_block([0, 0, 100, 100]), idx=1)
        smaller.setParentItem(text_layer)
        larger.setParentItem(text_layer)

        hit_item = scene.itemAt(QPointF(40, 40), QTransform())

        self.assertIs(hit_item, smaller)

    def test_auto_layout_resize_refreshes_box_stacking_priority(self):
        resized = TextBlkItem(make_text_block([0, 0, 10, 10]), idx=0)
        unchanged = TextBlkItem(make_text_block([0, 0, 20, 20]), idx=1)
        self.assertGreater(resized.zValue(), unchanged.zValue())

        resized.set_size(100, 100)

        self.assertLess(resized.zValue(), unchanged.zValue())


if __name__ == "__main__":
    unittest.main()
