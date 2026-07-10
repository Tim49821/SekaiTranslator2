import tempfile
import unittest

import numpy as np

from modules.ocr.ocr_ppv6_onnx import PaddleOCRv6ONNX, _write_ppocr_dict_from_yaml
from modules.ocr.base import OCR


class PaddleOCRv6ONNXTest(unittest.TestCase):
    def test_module_import_registers_ppv6_onnx(self):
        self.assertIs(OCR.module_dict["ppv6_onnx"], PaddleOCRv6ONNX)

    def test_pad_to_batch_keeps_empty_and_full_batches(self):
        self.assertEqual(PaddleOCRv6ONNX._pad_to_batch([], 6), [])
        crop = np.ones((2, 3, 3), dtype=np.uint8)
        batch = [crop.copy(), crop.copy()]

        padded = PaddleOCRv6ONNX._pad_to_batch(batch, 2)

        self.assertEqual(len(padded), 2)
        self.assertIs(padded, batch)

    def test_pad_to_batch_adds_blank_crops(self):
        crop = np.ones((2, 3, 3), dtype=np.uint8)

        padded = PaddleOCRv6ONNX._pad_to_batch([crop], 4)

        self.assertEqual(len(padded), 4)
        np.testing.assert_array_equal(padded[0], crop)
        np.testing.assert_array_equal(padded[1], np.zeros_like(crop))

    def test_write_ppocr_dict_from_yaml_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = f"{temp_dir}/inference.yml"
            dict_path = f"{temp_dir}/dict.txt"
            with open(yaml_path, "w", encoding="utf8") as f:
                f.write("PostProcess:\n  character_dict: ['a', '¢', null]\n")

            self.assertTrue(_write_ppocr_dict_from_yaml(yaml_path, dict_path))

            with open(dict_path, encoding="utf8") as f:
                self.assertEqual(f.read(), "a\n¢\n")


if __name__ == "__main__":
    unittest.main()
