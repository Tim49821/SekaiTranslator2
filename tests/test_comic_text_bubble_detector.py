import unittest
from types import SimpleNamespace

import numpy as np
import torch

from modules.textdetector.base import TEXTDETECTORS
from modules.textdetector.detector_comic_text_bubble import (
    HF_MODEL_REPO_ID,
    MODEL_PATH,
    ComicTextBubbleDetector,
)


class FakeProcessor:
    def __init__(self, result):
        self.result = result
        self.last_threshold = None
        self.last_target_sizes = None

    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.zeros((1, 3, 32, 32), dtype=torch.float32)}

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        self.last_threshold = threshold
        self.last_target_sizes = target_sizes
        return [self.result]


class FakeModel:
    device = torch.device('cpu')

    def __init__(self):
        self.config = SimpleNamespace(id2label={0: 'bubble', 1: 'text_bubble', 2: 'text_free'})

    def __call__(self, **inputs):
        return SimpleNamespace()


class ComicTextBubbleDetectorTest(unittest.TestCase):
    def make_detector(self, result):
        detector = ComicTextBubbleDetector(**{'mask dilate size': 0})
        detector.model = FakeModel()
        detector.processor = FakeProcessor(result)
        return detector

    def test_detector_is_registered_with_hf_snapshot_metadata(self):
        self.assertIs(TEXTDETECTORS.get('comic_text_bubble'), ComicTextBubbleDetector)
        self.assertEqual(ComicTextBubbleDetector.hf_model_repo_id, HF_MODEL_REPO_ID)
        self.assertEqual(ComicTextBubbleDetector.hf_model_save_dir, MODEL_PATH)
        self.assertEqual(
            ComicTextBubbleDetector.hf_model_required_files,
            ['config.json', 'model.safetensors', 'preprocessor_config.json'],
        )
        self.assertEqual(
            ComicTextBubbleDetector.hf_model_allow_patterns,
            ['config.json', 'model.safetensors', 'preprocessor_config.json'],
        )

    def test_default_detection_filters_bubble_and_keeps_text_labels(self):
        result = {
            'boxes': torch.tensor(
                [
                    [1, 1, 5, 5],
                    [10, 10, 20, 20],
                    [30, 10, 40, 20],
                ],
                dtype=torch.float32,
            ),
            'labels': torch.tensor([0, 1, 2], dtype=torch.int64),
        }
        detector = self.make_detector(result)

        mask, blk_list = detector._detect(np.zeros((50, 50, 3), dtype=np.uint8))

        self.assertEqual([blk.label for blk in blk_list], ['text_bubble', 'text_free'])
        self.assertEqual(mask[3, 3], 0)
        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(mask[15, 35], 255)
        self.assertEqual(detector.processor.last_threshold, 0.3)
        self.assertEqual(detector.processor.last_target_sizes.tolist(), [[50, 50]])

    def test_label_selection_can_include_bubble(self):
        result = {
            'boxes': torch.tensor([[1, 1, 5, 5]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.int64),
        }
        detector = self.make_detector(result)
        detector.set_param_value(
            'label',
            {'bubble': True, 'text_bubble': False, 'text_free': False},
        )

        mask, blk_list = detector._detect(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(len(blk_list), 1)
        self.assertEqual(blk_list[0].label, 'bubble')
        self.assertEqual(mask[3, 3], 255)


if __name__ == '__main__':
    unittest.main()
