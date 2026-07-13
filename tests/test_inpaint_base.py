import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from modules.inpaint.base import InpainterBase, filter_mask_by_bboxes
from modules.inpaint.inpaint_sdxl import SDXLInpainter
from utils.textblock import TextBlock
from utils.textblock_mask import (
    canny_flood,
    canny_flood_natural,
    feather_inpaint_result,
    refine_inpaint_mask_quality,
)


class CountingInpainter(InpainterBase):
    check_need_inpaint = False

    def __init__(self, **params):
        super().__init__(**params)
        self.calls = 0
        self.call_shapes = []

    def _inpaint(self, img, mask, textblock_list=None):
        self.calls += 1
        self.call_shapes.append(img.shape[:2])
        result = img.copy()
        result[mask > 0] = np.array([10, 20, 30], dtype=np.uint8)
        return result

    def moveToDevice(self, device: str, precision: str = None):
        return None


class FakeSDXLPipeline:
    def __init__(self):
        self.calls = []
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        image = np.array(kwargs["image"].convert("RGB"))
        mask = np.array(kwargs["mask_image"])
        self.calls.append(kwargs)
        output = np.zeros_like(image)
        output[:] = np.array([201, 31, 47], dtype=np.uint8)
        return SimpleNamespace(images=[Image.fromarray(output)])


class InpainterBaseTest(unittest.TestCase):
    def test_method3_reuses_method1_mask_and_forces_real_inpainting(self):
        img = np.full((120, 180, 3), 255, dtype=np.uint8)
        cv2.putText(
            img,
            "TEST",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        original = img.copy()

        method1_mask, method1_balloon, method1_info = canny_flood(img)
        method3_mask, method3_balloon, method3_info = canny_flood_natural(img)

        self.assertFalse(method1_info["need_inpaint"])
        self.assertTrue(method3_info["force_inpaint"])
        self.assertEqual(method3_info["context_ratio"], 2.5)
        self.assertEqual(method3_info["feather_radius"], 2)
        np.testing.assert_array_equal(method3_mask, method1_mask)
        np.testing.assert_array_equal(method3_balloon, method1_balloon)
        np.testing.assert_array_equal(img, original)

    def test_method3_feather_keeps_core_and_softens_only_cleanup_margin(self):
        original = np.zeros((15, 15, 3), dtype=np.uint8)
        candidate = np.full_like(original, 200)
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[5:10, 5:10] = 255

        blended, effective_mask = feather_inpaint_result(original, candidate, mask, radius=2)

        self.assertTrue(np.all(blended[mask > 0] == 200))
        self.assertTrue(np.all(blended[effective_mask == 0] == 0))
        margin = (effective_mask > 0) & (mask == 0)
        margin_values = blended[..., 0][margin]
        self.assertTrue(np.any((margin_values > 0) & (margin_values < 200)))
        self.assertGreater(np.count_nonzero(effective_mask), np.count_nonzero(mask))

    def test_base_move_to_device_requires_implementation(self):
        with self.assertRaises(NotImplementedError):
            InpainterBase().moveToDevice("cpu")

    def test_inpaint_does_not_mutate_input_mask(self):
        inpainter = CountingInpainter()
        img = np.zeros((12, 12, 3), dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[4:8, 4:8] = 255
        original_mask = mask.copy()

        inpainter.inpaint(img, mask)

        np.testing.assert_array_equal(mask, original_mask)

    def test_empty_mask_skips_model_call(self):
        inpainter = CountingInpainter()
        img = np.full((8, 8, 3), 77, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)

        result = inpainter.inpaint(img, mask)

        self.assertEqual(inpainter.calls, 0)
        np.testing.assert_array_equal(result, img)

    def test_rgba_alpha_is_preserved_for_opaque_regions(self):
        inpainter = CountingInpainter()
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[..., :3] = 5
        img[..., 3] = 255
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255

        result = inpainter.inpaint(img, mask)

        self.assertEqual(result.shape[2], 4)
        np.testing.assert_array_equal(result[..., 3], img[..., 3])

    def test_adjacent_textblocks_are_inpainted_as_one_cluster(self):
        inpainter = CountingInpainter()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:28, 20:28] = 255
        mask[20:28, 30:38] = 255
        blocks = [TextBlock([20, 20, 28, 28]), TextBlock([30, 20, 38, 28])]

        inpainter.inpaint(img, mask, blocks)

        self.assertEqual(inpainter.calls, 1)

    def test_empty_source_textblock_mask_is_excluded_when_requested(self):
        inpainter = CountingInpainter()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:10, 4:10] = 255
        mask[4:10, 18:24] = 255
        blocks = [
            TextBlock([3, 3, 12, 12], text=[""]),
            TextBlock([17, 3, 26, 12], text=["source"]),
        ]

        result = inpainter.inpaint(img, mask, blocks, ignore_empty_textblocks=True)

        np.testing.assert_array_equal(result[6, 6], [0, 0, 0])
        np.testing.assert_array_equal(result[6, 20], [10, 20, 30])
        self.assertEqual(inpainter.calls, 1)

    def test_empty_source_textblock_mask_is_kept_by_default(self):
        inpainter = CountingInpainter()
        img = np.zeros((24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[6:12, 6:12] = 255
        blocks = [TextBlock([5, 5, 14, 14], text=[""])]

        result = inpainter.inpaint(img, mask, blocks)

        np.testing.assert_array_equal(result[8, 8], [10, 20, 30])
        self.assertEqual(inpainter.calls, 1)

    def test_filter_mask_by_bboxes_keeps_only_textblock_regions(self):
        mask = np.full((24, 24), 255, dtype=np.uint8)
        blocks = [TextBlock([6, 6, 12, 12])]

        filtered = filter_mask_by_bboxes(mask, blocks)

        self.assertEqual(filtered[8, 8], 255)
        self.assertEqual(filtered[0, 0], 0)
        self.assertEqual(filtered[23, 23], 0)

    def test_refine_inpaint_mask_quality_cleans_fills_and_limits_to_balloon(self):
        mask = np.zeros((14, 14), dtype=np.uint8)
        mask[1, 1] = 255
        mask[5:10, 5:10] = 255
        mask[7, 7] = 0
        balloon = np.zeros_like(mask)
        balloon[:, :12] = 255

        refined = refine_inpaint_mask_quality(mask, balloon_mask=balloon, max_dilate=2)

        self.assertEqual(refined[1, 1], 0)
        self.assertEqual(refined[7, 7], 255)
        self.assertEqual(refined[:, 13].sum(), 0)

    def test_sdxl_inpaint_uses_square_crop_resized_mask_and_masked_blend(self):
        inpainter = SDXLInpainter()
        inpainter.model = FakeSDXLPipeline()
        inpainter.set_param_value("inpaint_size", 64)
        inpainter.set_param_value("feather_radius", 0)
        inpainter.set_param_value("context_scale", 2.0)
        img = np.zeros((40, 50, 3), dtype=np.uint8)
        img[..., 1] = 100
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[12:18, 20:26] = 255

        result = inpainter._inpaint(img, mask)
        call = inpainter.model.calls[0]

        self.assertEqual(call["image"].size, (64, 64))
        self.assertEqual(call["mask_image"].size, (64, 64))
        self.assertEqual(np.array(call["mask_image"]).max(), 255)
        self.assertTrue(np.all(result[mask > 0] == np.array([201, 31, 47], dtype=np.uint8)))
        self.assertTrue(np.all(result[mask == 0] == img[mask == 0]))


if __name__ == "__main__":
    unittest.main()
