import os.path as osp
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .base import DEVICE_SELECTOR, ProjImgTrans, TextBlock, TextDetectorBase, register_textdetectors
from utils.textblock import examine_textblk, sort_regions, sort_pnts


MODEL_PATH = 'data/models/comic-text-and-bubble-detector'
HF_MODEL_REPO_ID = 'ogkalu/comic-text-and-bubble-detector'
DEFAULT_LABELS = {
    'text_bubble': True,
    'text_free': True,
    'bubble': False,
}


def _move_inputs_to_device(inputs, device: str):
    if hasattr(inputs, 'to'):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, 'to') else value
        for key, value in inputs.items()
    }


@register_textdetectors('comic_text_bubble')
class ComicTextBubbleDetector(TextDetectorBase):
    dependencies = ['torch', 'transformers==4.57.1', 'safetensors>=0.8.0rc0', 'huggingface_hub>=0.34.0']
    hf_model_repo_id = HF_MODEL_REPO_ID
    hf_model_save_dir = MODEL_PATH
    hf_model_required_files = ['config.json', 'model.safetensors', 'preprocessor_config.json']
    hf_model_allow_patterns = ['config.json', 'model.safetensors', 'preprocessor_config.json']
    hf_model_download_on_prepare = True

    params = {
        'description': 'RT-DETR-v2 comic text and speech bubble detector.',
        'confidence threshold': {
            'display_name': '置信度阈值',
            'type': 'line_editor',
            'value': 0.3,
        },
        'font size multiplier': {
            'display_name': '字号乘数',
            'type': 'line_editor',
            'value': 1.,
        },
        'font size max': {
            'display_name': '最大字号',
            'type': 'line_editor',
            'value': -1,
        },
        'font size min': {
            'display_name': '最小字号',
            'type': 'line_editor',
            'value': -1,
        },
        'device': {
            **DEVICE_SELECTOR(),
            'display_name': '设备',
        },
        'label': {
            'value': DEFAULT_LABELS.copy(),
            'type': 'check_group',
            'display_name': '标签',
        },
        'mask dilate size': {
            'display_name': '掩码扩张尺寸',
            'type': 'line_editor',
            'value': 2,
        },
    }

    _load_model_keys = {'model', 'processor'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None
        self.processor = None

    def _missing_required_files(self) -> List[str]:
        return [
            filename
            for filename in self.hf_model_required_files
            if not osp.exists(osp.join(self.hf_model_save_dir, filename))
        ]

    def _load_model(self):
        missing_files = self._missing_required_files()
        if missing_files:
            raise FileNotFoundError(
                f'Comic text bubble detector model files are missing from {self.hf_model_save_dir}: '
                f'{", ".join(missing_files)}. Download {self.hf_model_repo_id} and place the snapshot there.'
            )

        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_save_dir)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(self.hf_model_save_dir)
        self.model.to(self.get_param_value('device')).eval()

    def get_valid_labels(self) -> List[str]:
        return [label for label, enabled in self.get_param_value('label').items() if enabled]

    def _label_name(self, label_id: int) -> str:
        id2label = getattr(self.model.config, 'id2label', {})
        return id2label.get(label_id) or id2label.get(str(label_id)) or str(label_id)

    def _box_to_textblock(self, xyxy: np.ndarray, label: str, im_w: int, im_h: int) -> Optional[TextBlock]:
        x1, y1, x2, y2 = xyxy.round().astype(np.int32).tolist()
        x1, x2 = sorted((int(np.clip(x1, 0, im_w)), int(np.clip(x2, 0, im_w))))
        y1, y2 = sorted((int(np.clip(y1, 0, im_h)), int(np.clip(y2, 0, im_h))))
        if x2 <= x1 or y2 <= y1:
            return None

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        pts_sorted, is_vertical = sort_pnts(pts)
        blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=label)
        blk.vertical = is_vertical
        blk.adjust_bbox()
        examine_textblk(blk, im_w, im_h)
        if blk._detected_font_size <= 0:
            blk._detected_font_size = blk.font_size
        return blk

    def _apply_font_size_params(self, blk_list: List[TextBlock]) -> None:
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk.detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        inputs = self.processor(images=img, return_tensors='pt')
        inputs = _move_inputs_to_device(inputs, self.model.device)

        target_sizes = torch.tensor([[im_h, im_w]], device=self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        result = self.processor.post_process_object_detection(
            outputs,
            threshold=self.get_param_value('confidence threshold'),
            target_sizes=target_sizes,
        )[0]

        valid_labels = set(self.get_valid_labels())
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        blk_list = []

        boxes = result.get('boxes', [])
        labels = result.get('labels', [])
        for box, label_id in zip(boxes, labels):
            if hasattr(label_id, 'detach'):
                label_id = int(label_id.detach().cpu().item())
            else:
                label_id = int(label_id)
            label_name = self._label_name(label_id)
            if label_name not in valid_labels:
                continue

            box_np = box.detach().cpu().numpy() if hasattr(box, 'detach') else np.asarray(box)
            blk = self._box_to_textblock(box_np, label_name, im_w, im_h)
            if blk is None:
                continue

            x1, y1, x2, y2 = blk.xyxy
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            blk_list.append(blk)

        blk_list = sort_regions(blk_list)
        self._apply_font_size_params(blk_list)

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device' and self.model is not None:
            self.model.to(self.get_param_value('device'))
