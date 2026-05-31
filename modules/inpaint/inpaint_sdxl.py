import os.path as osp
from typing import Tuple

import cv2
import numpy as np
import torch

from ..base import DEVICE_SELECTOR, soft_empty_cache
from .base import InpainterBase, register_inpainter


SDXL_INPAINT_REPO_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_PROMPT = "clean manga background, screen tone, line art, no text, no letters, consistent with surrounding panel"
DEFAULT_NEGATIVE_PROMPT = "text, letters, watermark, logo, blurry, color shift, distorted line art"


def _copy_make_border_reflect(img: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    border_type = cv2.BORDER_REFLECT if img.shape[0] > 1 and img.shape[1] > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(img, top, bottom, left, right, border_type)


def _resize_for_inpaint(img: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if img.shape[0] < size or img.shape[1] < size:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interpolation)


def _square_context_crop(img: np.ndarray, mask: np.ndarray, context_scale: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    nonzero = cv2.findNonZero((mask > 0).astype(np.uint8))
    if nonzero is None:
        return None, None, None, None

    im_h, im_w = img.shape[:2]
    x, y, w, h = cv2.boundingRect(nonzero)
    cx = x + w / 2
    cy = y + h / 2
    side = int(round(max(w, h) * context_scale))
    side = max(side, 256)
    side = max(side, w, h)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(im_w, x2)
    src_y2 = min(im_h, y2)

    pad_left = src_x1 - x1
    pad_top = src_y1 - y1
    pad_right = x2 - src_x2
    pad_bottom = y2 - src_y2

    crop_img = img[src_y1:src_y2, src_x1:src_x2]
    crop_mask = mask[src_y1:src_y2, src_x1:src_x2]
    crop_img = _copy_make_border_reflect(crop_img, pad_top, pad_bottom, pad_left, pad_right)
    crop_mask = cv2.copyMakeBorder(crop_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    source_rect = (src_x1, src_y1, src_x2, src_y2)
    source_slice_in_crop = (
        pad_left,
        pad_top,
        pad_left + src_x2 - src_x1,
        pad_top + src_y2 - src_y1,
    )
    return crop_img, crop_mask, source_rect, source_slice_in_crop


def _blend_masked_region(base: np.ndarray, generated: np.ndarray, mask: np.ndarray, feather_radius: int) -> np.ndarray:
    if feather_radius <= 0:
        alpha = (mask > 0).astype(np.float32)
    else:
        hard = (mask > 0).astype(np.float32)
        kernel_size = feather_radius * 2 + 1
        alpha = cv2.GaussianBlur(hard, (kernel_size, kernel_size), 0)
    alpha = np.clip(alpha[:, :, None], 0.0, 1.0)
    blended = generated.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


@register_inpainter("sdxl_inpaint")
class SDXLInpainter(InpainterBase):
    params = {
        "model_dir": {
            "type": "line_editor",
            "value": "data/models/sdxl_inpaint",
            "data_type": str,
            "description": "Local Diffusers snapshot directory for SDXL inpainting.",
        },
        "inpaint_size": {
            "type": "selector",
            "options": [768, 1024],
            "value": 1024,
        },
        "num_inference_steps": {
            "type": "line_editor",
            "value": 30,
            "data_type": int,
        },
        "strength": {
            "type": "line_editor",
            "value": 0.99,
            "data_type": float,
        },
        "guidance_scale": {
            "type": "line_editor",
            "value": 8.0,
            "data_type": float,
        },
        "seed": {
            "type": "line_editor",
            "value": 0,
            "data_type": int,
        },
        "prompt": {
            "type": "editor",
            "value": DEFAULT_PROMPT,
            "data_type": str,
        },
        "negative_prompt": {
            "type": "editor",
            "value": DEFAULT_NEGATIVE_PROMPT,
            "data_type": str,
        },
        "device": DEVICE_SELECTOR(not_supported=["privateuseone", "xpu"]),
        "context_scale": {
            "type": "line_editor",
            "value": 3.0,
            "data_type": float,
            "hidden": True,
        },
        "feather_radius": {
            "type": "line_editor",
            "value": 3,
            "data_type": int,
            "hidden": True,
        },
        "description": "High-quality local SDXL inpainting. Slower and heavier than LaMa; best used as an opt-in quality mode.",
    }

    _load_model_keys = {"model"}
    hf_model_repo_id = SDXL_INPAINT_REPO_ID
    hf_model_save_dir = "data/models/sdxl_inpaint"
    hf_model_required_files = ["model_index.json"]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params["device"]["value"]
        self.model = None

    def _load_model(self):
        try:
            from diffusers import StableDiffusionXLInpaintPipeline
        except Exception as exc:
            raise ImportError(
                "sdxl_inpaint requires diffusers, accelerate, and safetensors. "
                "Install the updated requirements before selecting this inpainter."
            ) from exc

        model_dir = self.get_param_value("model_dir")
        if not osp.exists(osp.join(model_dir, "model_index.json")):
            raise FileNotFoundError(
                f"SDXL inpaint model files are missing from {model_dir}. "
                f"Download {SDXL_INPAINT_REPO_ID} there, or run prepare with "
                "BALLOONTRANS_DOWNLOAD_HF_MODEL_ON_PREPARE=true."
            )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
            add_watermarker=False,
        )
        self.model.set_progress_bar_config(disable=True)
        self.moveToDevice(self.device)

    def moveToDevice(self, device: str, precision: str = None):
        if self.model is not None:
            self.model.to(device)
        self.device = device

    def unload_model(self, empty_cache=False):
        unloaded = super().unload_model(empty_cache=False)
        if empty_cache or unloaded:
            soft_empty_cache()
        return unloaded

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "device":
            self.device = self.params["device"]["value"]
            if self.model is not None:
                self.model.to(self.device)

    @torch.inference_mode()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None) -> np.ndarray:
        if not np.any(mask > 0):
            return img.copy()

        crop_img, crop_mask, source_rect, source_slice = _square_context_crop(
            img,
            (mask > 0).astype(np.uint8) * 255,
            context_scale=max(float(self.get_param_value("context_scale")), 1.0),
        )
        if crop_img is None:
            return img.copy()

        from PIL import Image

        inpaint_size = int(self.get_param_value("inpaint_size"))
        inpaint_size = max(64, (inpaint_size // 8) * 8)
        pipe_img = _resize_for_inpaint(crop_img, inpaint_size, is_mask=False)
        pipe_mask = _resize_for_inpaint(crop_mask, inpaint_size, is_mask=True)

        seed = int(self.get_param_value("seed"))
        generator = None
        if seed >= 0:
            generator_device = self.device if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)

        output = self.model(
            prompt=self.get_param_value("prompt"),
            negative_prompt=self.get_param_value("negative_prompt"),
            image=Image.fromarray(pipe_img),
            mask_image=Image.fromarray(pipe_mask),
            height=inpaint_size,
            width=inpaint_size,
            num_inference_steps=int(self.get_param_value("num_inference_steps")),
            strength=float(self.get_param_value("strength")),
            guidance_scale=float(self.get_param_value("guidance_scale")),
            generator=generator,
        )
        out_pil = output.images[0]
        generated = np.array(out_pil.convert("RGB"))
        generated = cv2.resize(generated, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        sx1, sy1, sx2, sy2 = source_rect
        cx1, cy1, cx2, cy2 = source_slice
        generated_src = generated[cy1:cy2, cx1:cx2]
        mask_src = crop_mask[cy1:cy2, cx1:cx2]

        result = img.copy()
        base_src = result[sy1:sy2, sx1:sx2]
        feather_radius = int(self.get_param_value("feather_radius"))
        result[sy1:sy2, sx1:sx2] = _blend_masked_region(base_src, generated_src, mask_src, feather_radius)
        return result
