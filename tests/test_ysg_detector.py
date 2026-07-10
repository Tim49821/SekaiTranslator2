import unittest

from modules.textdetector.detector_ysg import YSGYoloDetector, update_ckpt_list


class YSGYoloDetectorTest(unittest.TestCase):
    def test_download_file_list_contains_yolo26_models(self):
        files = {item["files"] for item in YSGYoloDetector.download_file_list}

        self.assertIn("data/models/ysgyolo_yolo26_2.0.pt", files)
        self.assertIn("data/models/ysgyolo_yolo26OBB_2.0.pt", files)

    def test_update_ckpt_list_includes_downloadable_defaults(self):
        ckpts = update_ckpt_list()

        self.assertIn("data/models/ysgyolo_yolo26_2.0.pt", ckpts)
        self.assertIn("data/models/ysgyolo_yolo26OBB_2.0.pt", ckpts)


if __name__ == "__main__":
    unittest.main()
