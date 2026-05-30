import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from modules.textdetector.base import TEXTDETECTORS
from modules.textdetector.detector_paddlex_ppocrv5 import (
    PPOCRv5MobileDetector,
    PPOCRv5ServerDetector,
)


class FakePaddleXResult:
    def __init__(self, detection_res):
        self._detection_res = detection_res

    @property
    def json(self):
        return {"res": self._detection_res}


class FakePaddleXModel:
    def __init__(self, detection_res):
        self.detection_res = detection_res
        self.last_img_shape = None
        self.last_batch_size = None
        self.closed = False

    def predict(self, img, batch_size=1):
        self.last_img_shape = img.shape
        self.last_batch_size = batch_size
        return [FakePaddleXResult(self.detection_res)]

    def close(self):
        self.closed = True


class FakeCreateModel:
    last_kwargs = None

    def __call__(self, **kwargs):
        type(self).last_kwargs = kwargs
        return FakePaddleXModel({"dt_polys": [], "dt_scores": []})


class PaddleXPPocrV5DetectorTest(unittest.TestCase):
    def make_detector(self, detection_res, detector_cls=PPOCRv5MobileDetector, **params):
        detector = detector_cls(**{"mask dilate size": 0, **params})
        detector.model = FakePaddleXModel(detection_res)
        return detector

    def rect_poly(self, x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_detectors_are_registered_with_official_model_names(self):
        self.assertIs(TEXTDETECTORS.get("PP-OCRv5_server_det"), PPOCRv5ServerDetector)
        self.assertIs(TEXTDETECTORS.get("PP-OCRv5_mobile_det"), PPOCRv5MobileDetector)

    def test_load_model_passes_paddlex_create_model_kwargs(self):
        fake_create_model = FakeCreateModel()
        fake_paddlex = ModuleType("paddlex")
        fake_paddlex.create_model = fake_create_model

        with patch.dict(sys.modules, {"paddlex": fake_paddlex}):
            with patch(
                "modules.textdetector.detector_paddlex_ppocrv5.PADDLE_RUNTIME_AVAILABLE",
                True,
            ):
                with patch(
                    "modules.textdetector.detector_paddlex_ppocrv5.PADDLEX_AVAILABLE",
                    True,
                ):
                    detector = PPOCRv5ServerDetector(
                        **{
                            "device": "cpu",
                            "model dir": "data/models/custom-det",
                            "model source": "bos",
                            "limit side len": "1280",
                            "limit type": "max",
                            "thresh": "0.2",
                            "box thresh": "0.6",
                            "unclip ratio": "1.7",
                            "engine": "paddle",
                            "use_hpip": True,
                        }
                    )
                    before_source = os.environ.get("PADDLE_PDX_MODEL_SOURCE")
                    detector._load_model()

        self.assertEqual(
            FakeCreateModel.last_kwargs,
            {
                "model_name": "PP-OCRv5_server_det",
                "device": "cpu",
                "use_hpip": True,
                "model_dir": "data/models/custom-det",
                "limit_side_len": 1280,
                "limit_type": "max",
                "thresh": 0.2,
                "box_thresh": 0.6,
                "unclip_ratio": 1.7,
                "engine": "paddle",
            },
        )
        self.assertEqual(os.environ.get("PADDLE_PDX_MODEL_SOURCE"), before_source)

    def test_detection_polygons_are_converted_to_textblocks_and_mask(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        [[10, 10], [40, 10], [40, 30], [10, 30]],
                        [[60, 10], [90, 10], [90, 30], [60, 30]],
                    ],
                    dtype=np.int16,
                ),
                "dt_scores": np.array([0.8, 0.2], dtype=np.float32),
            },
            **{"score threshold": 0.5, "batch size": 3},
        )

        mask, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 1)
        self.assertEqual(blk_list[0].label, "PP-OCRv5_mobile_det")
        self.assertEqual(blk_list[0].xyxy, [10, 10, 40, 30])
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[20, 70], 0)
        self.assertEqual(detector.model.last_batch_size, 3)

    def test_nearby_boxes_are_merged_into_one_textblock(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        self.rect_poly(10, 10, 30, 30),
                        self.rect_poly(38, 10, 58, 30),
                    ],
                    dtype=np.int16,
                )
            }
        )

        mask, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 1)
        self.assertEqual(len(blk_list[0].lines), 2)
        self.assertEqual(blk_list[0].xyxy, [10, 10, 58, 30])
        self.assertEqual(mask[20, 15], 255)
        self.assertEqual(mask[20, 45], 255)
        self.assertEqual(mask[20, 34], 0)

    def test_far_boxes_are_not_merged(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        self.rect_poly(10, 10, 30, 30),
                        self.rect_poly(70, 10, 90, 30),
                    ],
                    dtype=np.int16,
                )
            }
        )

        _, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 2)
        self.assertEqual([blk.xyxy for blk in blk_list], [[10, 10, 30, 30], [70, 10, 90, 30]])

    def test_merge_nearby_boxes_can_be_disabled(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        self.rect_poly(10, 10, 30, 30),
                        self.rect_poly(38, 10, 58, 30),
                    ],
                    dtype=np.int16,
                )
            },
            **{"merge nearby boxes": False},
        )

        _, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 2)
        self.assertEqual([len(blk.lines) for blk in blk_list], [1, 1])

    def test_boxes_with_different_direction_are_not_merged(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        self.rect_poly(10, 10, 30, 30),
                        self.rect_poly(34, 10, 50, 60),
                    ],
                    dtype=np.int16,
                )
            }
        )

        _, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 2)
        self.assertEqual({blk.vertical for blk in blk_list}, {False, True})

    def test_nearby_box_merge_is_transitive(self):
        detector = self.make_detector(
            {
                "dt_polys": np.array(
                    [
                        self.rect_poly(10, 10, 30, 30),
                        self.rect_poly(38, 10, 58, 30),
                        self.rect_poly(66, 10, 86, 30),
                    ],
                    dtype=np.int16,
                )
            }
        )

        _, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 1)
        self.assertEqual(len(blk_list[0].lines), 3)
        self.assertEqual(blk_list[0].xyxy, [10, 10, 86, 30])

    def test_flat_polygons_and_dict_results_are_supported(self):
        detector = PPOCRv5MobileDetector(**{"mask dilate size": 0})
        detector.model = type(
            "DictResultModel",
            (),
            {
                "predict": lambda self, img, batch_size=1: {
                    "res": {"dt_polys": [[10, 10, 40, 10, 40, 30, 10, 30]]}
                }
            },
        )()

        mask, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 1)
        self.assertEqual(blk_list[0].xyxy, [10, 10, 40, 30])
        self.assertEqual(mask[20, 20], 255)

    def test_invalid_polygons_are_ignored(self):
        detector = self.make_detector(
            {
                "dt_polys": [
                    [],
                    [[20, 20], [20, 20], [20, 20], [20, 20]],
                    [1, 2, 3],
                ],
                "dt_scores": [1.0, 1.0, 1.0],
            }
        )

        mask, blk_list = detector._detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(blk_list, [])
        self.assertFalse(mask.any())

    def test_model_close_is_called_on_unload(self):
        detector = self.make_detector({"dt_polys": [], "dt_scores": []})
        model = detector.model

        self.assertTrue(detector.unload_model())

        self.assertTrue(model.closed)
        self.assertIsNone(detector.model)


if __name__ == "__main__":
    unittest.main()
