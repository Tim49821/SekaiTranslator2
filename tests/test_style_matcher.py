import os
import sys
import unittest

import cv2
import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

from utils.config import ProgramConfig
from utils.fontformat import FontFormat, TextAlignment
from utils.style_matcher import apply_smart_style, extract_source_style
from utils.textblock import TextBlock


def color_distance(a, b):
    a = np.array(a[:3], dtype=np.float32)
    b = np.array(b[:3], dtype=np.float32)
    return float(np.linalg.norm(a - b))


def make_block(x1=10, y1=20, x2=250, y2=95, text='TEST'):
    return TextBlock(
        xyxy=[x1, y1, x2, y2],
        lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        text=[text],
        fontformat=FontFormat(font_size=54, frgb=[0, 0, 0], srgb=[255, 255, 255]),
        _detected_font_size=54,
    )


class StyleMatcherTest(unittest.TestCase):

    def test_extracts_black_fill_without_visible_stroke(self):
        img = np.full((130, 280, 3), 255, dtype=np.uint8)
        cv2.putText(img, 'TEST', (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_8)

        style = extract_source_style(img, make_block())

        self.assertGreater(style['confidence'], 0.2)
        self.assertLess(color_distance(style['fill_rgb'], [0, 0, 0]), 45)
        self.assertEqual(style['stroke_width_ratio'], 0.0)

    def test_extracts_colored_fill_and_stroke(self):
        img = np.full((150, 320, 3), [235, 235, 235], dtype=np.uint8)
        stroke_rgb = [20, 80, 220]
        fill_rgb = [245, 220, 30]
        cv2.putText(img, 'TEST', (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.0, stroke_rgb, 9, cv2.LINE_8)
        cv2.putText(img, 'TEST', (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.0, fill_rgb, 3, cv2.LINE_8)

        style = extract_source_style(img, make_block(10, 20, 300, 115))

        self.assertGreater(style['confidence'], 0.2)
        self.assertLess(color_distance(style['fill_rgb'], fill_rgb), 90)
        self.assertLess(color_distance(style['stroke_rgb'], stroke_rgb), 110)
        self.assertGreater(style['stroke_width_ratio'], 0.01)

    def test_smart_style_respects_global_override_flags(self):
        cfg = ProgramConfig()
        cfg.let_fntsize_flag = 1
        cfg.let_fntstroke_flag = 1
        cfg.let_fntcolor_flag = 1
        cfg.let_fnt_scolor_flag = 1
        cfg.let_alignment_flag = 1
        cfg.let_family_flag = 1

        blk = make_block()
        blk.font_size = 24
        blk.stroke_width = 0.0
        blk.alignment = int(TextAlignment.Left)
        blk.source_style = {
            'font_size_px': 72,
            'fill_rgb': [200, 10, 10],
            'stroke_rgb': [10, 200, 10],
            'stroke_width_ratio': 0.2,
            'alignment': int(TextAlignment.Center),
            'confidence': 1.0,
        }

        updated = apply_smart_style(blk, FontFormat(), cfg)

        self.assertEqual(updated, set())
        self.assertEqual(blk.font_size, 24)
        self.assertEqual(blk.stroke_width, 0.0)
        self.assertEqual(blk.alignment, int(TextAlignment.Left))


if __name__ == '__main__':
    unittest.main()
