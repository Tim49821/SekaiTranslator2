import unittest

import numpy as np

from utils.imgproc_utils import hex2bgr


class ImgprocUtilsTest(unittest.TestCase):
    def test_hex2bgr_keeps_low_bit_channels(self):
        color = np.array([0x123456], dtype=np.int64)

        bgr = hex2bgr(color)

        np.testing.assert_array_equal(bgr, np.array([[0x12, 0x34, 0x56]]))


if __name__ == "__main__":
    unittest.main()
