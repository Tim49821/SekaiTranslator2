import copy
import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import DEVICE_SELECTOR, ProjImgTrans, TextBlock, TextDetectorBase, register_textdetectors
from utils.textblock import examine_textblk, sort_regions, sort_pnts


PADDLE_OCR_PATH = os.path.join("data", "models", "paddle-ocr")
os.environ["PPOCR_HOME"] = PADDLE_OCR_PATH
os.environ["PADDLEOCR_HOME"] = PADDLE_OCR_PATH
os.environ["PADDLE_PDX_CACHE_HOME"] = PADDLE_OCR_PATH
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PADDLE_RUNTIME_AVAILABLE = importlib.util.find_spec("paddle") is not None
PADDLEX_AVAILABLE = importlib.util.find_spec("paddlex") is not None
PADDLE_RUNTIME_INSTALL_HINT = (
    "PP-OCRv5 PaddleX detector requires the PaddlePaddle runtime. Install "
    "'paddlepaddle>=3.3.0,<4.0.0', or the matching 'paddlepaddle-gpu' package for CUDA."
)
PADDLEX_INSTALL_HINT = (
    "PP-OCRv5 PaddleX detector requires PaddleX. Install it with "
    "'python -m pip install -U paddlex', or install 'paddleocr==3.5.0' which depends on PaddleX."
)


def _exception_chain_text(exc: BaseException) -> str:
    messages = []
    current = exc
    while current is not None:
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return " ".join(messages)


class PaddleXPPocrV5DetectorBase(TextDetectorBase):
    paddlex_model_name = ""

    params = {
        "description": "PaddleX PP-OCRv5 text detection model.",
        "device": {
            **DEVICE_SELECTOR(not_supported=["mps", "privateuseone"]),
            "display_name": "Device",
        },
        "model dir": {
            "type": "selector",
            "options": [""],
            "value": "",
            "editable": True,
            "path_selector": True,
            "size": "median",
            "description": "Optional local PaddleX inference model directory.",
        },
        "model source": {
            "type": "selector",
            "options": ["", "huggingface", "aistudio", "bos", "modelscope"],
            "value": "",
            "description": "Optional PADDLE_PDX_MODEL_SOURCE override for official model downloads.",
        },
        "limit side len": {
            "display_name": "Limit side len",
            "type": "line_editor",
            "value": "",
            "description": "Optional PaddleX limit_side_len. Empty uses the official model default.",
        },
        "limit type": {
            "display_name": "Limit type",
            "type": "selector",
            "options": ["", "max", "min"],
            "value": "",
            "description": "Optional PaddleX limit_type. Empty uses the official model default.",
        },
        "thresh": {
            "display_name": "Thresh",
            "type": "line_editor",
            "value": "",
            "description": "Optional text-pixel threshold. Empty uses the official model default.",
        },
        "box thresh": {
            "display_name": "Box thresh",
            "type": "line_editor",
            "value": "",
            "description": "Optional detected-box threshold. Empty uses the official model default.",
        },
        "unclip ratio": {
            "display_name": "Unclip ratio",
            "type": "line_editor",
            "value": "",
            "description": "Optional text-region expansion ratio. Empty uses the official model default.",
        },
        "engine": {
            "type": "selector",
            "options": ["", "paddle", "paddle_static", "paddle_dynamic", "hpi", "flexible"],
            "value": "",
            "description": "Optional PaddleX inference engine. Empty lets PaddleX choose.",
        },
        "use_hpip": {
            "display_name": "Use HPIP",
            "type": "checkbox",
            "value": False,
            "description": "Enable the PaddleX high-performance inference plugin.",
        },
        "batch size": {
            "display_name": "Batch size",
            "type": "line_editor",
            "value": 1,
        },
        "score threshold": {
            "display_name": "Score threshold",
            "type": "line_editor",
            "value": 0.0,
            "description": "Additional post-filter for dt_scores.",
        },
        "font size multiplier": {
            "display_name": "Font size multiplier",
            "type": "line_editor",
            "value": 1.0,
        },
        "font size max": {
            "display_name": "Font size max",
            "type": "line_editor",
            "value": -1,
        },
        "font size min": {
            "display_name": "Font size min",
            "type": "line_editor",
            "value": -1,
        },
        "mask dilate size": {
            "display_name": "Mask dilate size",
            "type": "line_editor",
            "value": 2,
        },
    }

    _load_model_keys = {"model"}
    _reload_param_keys = {
        "device",
        "model dir",
        "model source",
        "limit side len",
        "limit type",
        "thresh",
        "box thresh",
        "unclip ratio",
        "engine",
        "use_hpip",
    }

    def __init__(self, **params) -> None:
        self.params = copy.deepcopy(self.__class__.params)
        super().__init__(**params)
        self.model = None

    def _paddle_device(self) -> str:
        device = self.get_param_value("device")
        if device == "cuda":
            return "gpu:0"
        if device.startswith("cuda:"):
            return f"gpu:{device.split(':', 1)[1]}"
        if device == "cpu" or device.startswith(("gpu", "xpu", "npu")):
            return device if ":" in device or device == "cpu" else f"{device}:0"
        self.logger.warning(
            "PaddleX PP-OCRv5 detector does not support this UI device selector. "
            "Falling back to CPU for device: %s",
            device,
        )
        return "cpu"

    def _optional_int_param(self, key: str) -> Optional[int]:
        raw = self.get_param_value(key)
        if raw is None or str(raw).strip() == "":
            return None
        return int(raw)

    def _optional_float_param(self, key: str) -> Optional[float]:
        raw = self.get_param_value(key)
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)

    def _create_model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_name": self.paddlex_model_name,
            "device": self._paddle_device(),
            "use_hpip": self.get_param_value("use_hpip"),
        }

        model_dir = self.get_param_value("model dir").strip()
        if model_dir:
            kwargs["model_dir"] = model_dir

        limit_side_len = self._optional_int_param("limit side len")
        if limit_side_len is not None:
            kwargs["limit_side_len"] = limit_side_len

        limit_type = self.get_param_value("limit type")
        if limit_type:
            kwargs["limit_type"] = limit_type

        for param_key, kwarg_key in (
            ("thresh", "thresh"),
            ("box thresh", "box_thresh"),
            ("unclip ratio", "unclip_ratio"),
        ):
            value = self._optional_float_param(param_key)
            if value is not None:
                kwargs[kwarg_key] = value

        engine = self.get_param_value("engine")
        if engine:
            kwargs["engine"] = engine

        return kwargs

    def _load_model(self):
        if self.model is not None:
            return
        if not PADDLE_RUNTIME_AVAILABLE:
            raise RuntimeError(PADDLE_RUNTIME_INSTALL_HINT)
        if not PADDLEX_AVAILABLE:
            raise RuntimeError(PADDLEX_INSTALL_HINT)

        try:
            from paddlex import create_model
        except ImportError as exc:
            raise RuntimeError(PADDLEX_INSTALL_HINT) from exc

        model_source = self.get_param_value("model source").strip()
        previous_model_source = os.environ.get("PADDLE_PDX_MODEL_SOURCE")
        if model_source:
            os.environ["PADDLE_PDX_MODEL_SOURCE"] = model_source

        try:
            self.model = create_model(**self._create_model_kwargs())
        except Exception as exc:
            detail = _exception_chain_text(exc)
            raise RuntimeError(
                f"Failed to load {self.paddlex_model_name} detector. Original error: {detail}"
            ) from exc
        finally:
            if model_source:
                if previous_model_source is None:
                    os.environ.pop("PADDLE_PDX_MODEL_SOURCE", None)
                else:
                    os.environ["PADDLE_PDX_MODEL_SOURCE"] = previous_model_source

    def _predict(self, img: np.ndarray):
        result = self.model.predict(img, batch_size=int(self.get_param_value("batch size")))
        if (
            not isinstance(result, (list, tuple))
            and hasattr(result, "__iter__")
            and not isinstance(result, dict)
            and not hasattr(result, "json")
        ):
            result = list(result)
        return result

    def _result_to_dict(self, result: Any) -> Dict:
        if hasattr(result, "json"):
            data = result.json
            if callable(data):
                data = data()
        else:
            data = result

        if isinstance(data, dict) and isinstance(data.get("res"), dict):
            data = data["res"]
        return data if isinstance(data, dict) else {}

    def _extract_detection_result(self, output: Any) -> Dict:
        if output is None:
            return {}
        if not isinstance(output, (list, tuple)):
            output = [output]
        if not output:
            return {}
        return self._result_to_dict(output[0])

    def _get_indexed_value(self, values: Any, index: int):
        if values is None:
            return None
        try:
            if len(values) <= index:
                return None
            return values[index]
        except TypeError:
            return None

    def _poly_to_points(self, poly: Any, im_w: int, im_h: int) -> Optional[np.ndarray]:
        try:
            pts = np.asarray(poly, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if pts.ndim == 1:
            if pts.size == 4:
                x1, y1, x2, y2 = pts.tolist()
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            elif pts.size >= 8 and pts.size % 2 == 0:
                pts = pts.reshape((-1, 2))
            else:
                return None
        elif pts.ndim == 2 and pts.shape[1] >= 2:
            pts = pts[:, :2]
        else:
            return None

        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 4:
            return None

        pts[:, 0] = np.clip(pts[:, 0], 0, im_w)
        pts[:, 1] = np.clip(pts[:, 1], 0, im_h)

        if (pts[:, 0].max() - pts[:, 0].min()) < 1 or (pts[:, 1].max() - pts[:, 1].min()) < 1:
            return None

        return pts

    def _poly_to_textblock(self, pts: np.ndarray, im_w: int, im_h: int) -> Optional[TextBlock]:
        if pts.shape[0] == 4:
            quad = pts
        else:
            quad = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))

        try:
            pts_sorted, is_vertical = sort_pnts(quad.astype(np.float32))
        except AssertionError:
            return None

        blk = TextBlock(
            lines=[pts_sorted.astype(np.int32)],
            src_is_vertical=is_vertical,
            label=self.paddlex_model_name,
        )
        blk.vertical = is_vertical
        blk.adjust_bbox()
        examine_textblk(blk, im_w, im_h)
        if blk._detected_font_size <= 0:
            blk._detected_font_size = blk.font_size
        return blk

    def _apply_font_size_params(self, blk_list: List[TextBlock]) -> None:
        fnt_rsz = self.get_param_value("font size multiplier")
        fnt_max = self.get_param_value("font size max")
        fnt_min = self.get_param_value("font size min")
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
        result = self._extract_detection_result(self._predict(img))
        dt_polys = result.get("dt_polys")
        if dt_polys is None:
            dt_polys = result.get("rec_polys")
        if dt_polys is None:
            dt_polys = []

        dt_scores = result.get("dt_scores")
        if dt_scores is None:
            dt_scores = result.get("rec_scores")
        score_threshold = float(self.get_param_value("score threshold"))

        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        blk_list = []
        for idx, poly in enumerate(dt_polys):
            score = self._get_indexed_value(dt_scores, idx)
            if score is not None and float(score) < score_threshold:
                continue

            pts = self._poly_to_points(poly, im_w, im_h)
            if pts is None:
                continue

            blk = self._poly_to_textblock(pts, im_w, im_h)
            if blk is None:
                continue

            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
            blk_list.append(blk)

        blk_list = sort_regions(blk_list)
        self._apply_font_size_params(blk_list)

        ksize = self.get_param_value("mask dilate size")
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def unload_model(self, empty_cache=False):
        if self.model is not None and hasattr(self.model, "close"):
            try:
                self.model.close()
            except Exception:
                self.logger.debug("Failed to close PaddleX PP-OCRv5 detector.", exc_info=True)
        return super().unload_model(empty_cache=empty_cache)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in self._reload_param_keys and self.model is not None:
            self.unload_model(empty_cache=True)


@register_textdetectors("PP-OCRv5_server_det")
class PPOCRv5ServerDetector(PaddleXPPocrV5DetectorBase):
    paddlex_model_name = "PP-OCRv5_server_det"


@register_textdetectors("PP-OCRv5_mobile_det")
class PPOCRv5MobileDetector(PaddleXPPocrV5DetectorBase):
    paddlex_model_name = "PP-OCRv5_mobile_det"
