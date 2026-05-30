import unittest

from modules.inpaint.base import InpainterBase


class InpainterBaseTest(unittest.TestCase):
    def test_base_move_to_device_requires_implementation(self):
        with self.assertRaises(NotImplementedError):
            InpainterBase().moveToDevice("cpu")


if __name__ == "__main__":
    unittest.main()
