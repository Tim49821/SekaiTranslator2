import numpy as np
from typing import List
import os
import logging
import importlib.util

LOGGER = logging.getLogger("BallonTranslator")

# Specify the path for storing PaddleOCR models before importing PaddleOCR.
PADDLE_OCR_PATH = os.path.join("data", "models", "paddle-ocr")
os.environ["PPOCR_HOME"] = PADDLE_OCR_PATH
os.environ["PADDLEOCR_HOME"] = PADDLE_OCR_PATH
os.environ["PADDLE_PDX_CACHE_HOME"] = PADDLE_OCR_PATH
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from paddleocr import PaddleOCR

    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    LOGGER.debug(
        "PaddleOCR is not installed, so the module will not be initialized. \nCheck this issue https://github.com/dmMaze/BallonsTranslator/issues/835#issuecomment-2772940806"
    )

PADDLE_RUNTIME_AVAILABLE = importlib.util.find_spec("paddle") is not None
PADDLE_RUNTIME_INSTALL_HINT = (
    "PaddleOCR requires the PaddlePaddle runtime for inference, but the "
    "'paddlepaddle' package is not installed. Install it with "
    "'python -m pip install paddlepaddle>=3.3.0,<4.0.0'. CUDA users can "
    "install the matching 'paddlepaddle-gpu' package instead."
)

import cv2
import re

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

if PADDLE_OCR_AVAILABLE:

    @register_OCR("paddle_ocr")
    class PaddleOCRModule(OCRBase):
        # Mapping language names to PaddleOCR codes
        lang_map = {
            "Chinese & English": "ch",
            "English": "en",
            "French": "fr",
            "German": "german",
            "Japanese": "japan",
            "Korean": "korean",
            "Chinese Traditional": "chinese_cht",
            "Italian": "it",
            "Spanish": "es",
            "Portuguese": "pt",
            "Russian": "ru",
            "Ukrainian": "uk",
            "Belarusian": "be",
            "Telugu": "te",
            "Saudi Arabia": "sa",
            "Tamil": "ta",
            "Afrikaans": "af",
            "Azerbaijani": "az",
            "Bosnian": "bs",
            "Czech": "cs",
            "Welsh": "cy",
            "Danish": "da",
            "Dutch": "nl",
            "Norwegian": "no",
            "Polish": "pl",
            "Romanian": "ro",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Albanian": "sq",
            "Swedish": "sv",
            "Swahili": "sw",
            "Tagalog": "tl",
            "Turkish": "tr",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Mongolian": "mn",
            "Arabic": "ar",
            "Hindi": "hi",
            "Uyghur": "ug",
            "Persian": "fa",
            "Urdu": "ur",
            "Serbian (Latin)": "rs_latin",
            "Occitan": "oc",
            "Marathi": "mr",
            "Nepali": "ne",
            "Serbian (Cyrillic)": "rs_cyrillic",
            "Bulgarian": "bg",
            "Estonian": "et",
            "Irish": "ga",
            "Croatian": "hr",
            "Hungarian": "hu",
            "Indonesian": "id",
            "Icelandic": "is",
            "Kurdish": "ku",
            "Lithuanian": "lt",
            "Latvian": "lv",
            "Maori": "mi",
            "Malay": "ms",
            "Maltese": "mt",
            "Adyghe": "ady",
            "Kabardian": "kbd",
            "Avar": "ava",
            "Dargwa": "dar",
            "Ingush": "inh",
            "Lak": "lbe",
            "Lezghian": "lez",
            "Tabassaran": "tab",
            "Bihari": "bh",
            "Maithili": "mai",
            "Angika": "ang",
            "Bhojpuri": "bho",
            "Magahi": "mah",
            "Nagpur": "sck",
            "Newari": "new",
            "Goan Konkani": "gom",
        }

        _load_model_keys = {"model"}
        _paddleocr_api_style = "new"

        params = {
            "language": {
                "type": "selector",
                "options": list(lang_map.keys()),
                "value": "English",  # Default language
                "description": "Select the language for OCR",
            },
            "device": DEVICE_SELECTOR(),
            "use_angle_cls": {
                "type": "checkbox",
                "value": False,
                "description": "Enable angle classification for rotated text",
            },
            "ocr_version": {
                "type": "selector",
                "options": ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"],
                "value": "PP-OCRv5",
                "description": "Select the OCR model version",
            },
            "enable_mkldnn": {
                "type": "checkbox",
                "value": False,
                "description": "Enable MKL-DNN for CPU acceleration",
            },
            "det_limit_side_len": {
                "value": 960,
                "description": "Maximum side length for text detection",
            },
            "rec_batch_num": {
                "value": 6,
                "description": "Batch size for text recognition",
            },
            "drop_score": {
                "value": 0.5,
                "description": "Confidence threshold for text recognition",
            },
            "text_case": {
                "type": "selector",
                "options": ["Uppercase", "Capitalize Sentences", "Lowercase"],
                "value": "Capitalize Sentences",
                "description": "Text case transformation",
            },
            "output_format": {
                "type": "selector",
                "options": ["Single Line", "As Recognized"],
                "value": "As Recognized",
                "description": "Text output format",
            },
        }

        device = DEFAULT_DEVICE

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.language = self.params["language"]["value"]
            self.device = self.params["device"]["value"]
            self.use_angle_cls = self.params["use_angle_cls"]["value"]
            self.ocr_version = self.params["ocr_version"]["value"]
            self.enable_mkldnn = self.params["enable_mkldnn"]["value"]
            self.det_limit_side_len = self.params["det_limit_side_len"]["value"]
            self.rec_batch_num = self.params["rec_batch_num"]["value"]
            self.drop_score = self.params["drop_score"]["value"]
            self.text_case = self.params["text_case"]["value"]
            self.output_format = self.params["output_format"]["value"]
            self.model = None
            self._setup_logging()

        def _setup_logging(self):
            if self.debug_mode:
                logging.getLogger("ppocr").setLevel(logging.DEBUG)
                logging.getLogger("paddleocr").setLevel(logging.DEBUG)
                logging.getLogger("predict_system").setLevel(logging.DEBUG)
            else:
                logging.getLogger("ppocr").setLevel(logging.WARNING)
                logging.getLogger("paddleocr").setLevel(logging.WARNING)
                logging.getLogger("predict_system").setLevel(logging.WARNING)

        def _build_new_init_kwargs(self, lang_code: str, use_gpu: bool):
            kwargs = {
                "lang": lang_code,
                "ocr_version": self.ocr_version,
                "device": "gpu:0" if use_gpu else "cpu",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": self.use_angle_cls,
                "text_det_limit_side_len": self.det_limit_side_len,
                "text_recognition_batch_size": self.rec_batch_num,
                "text_rec_score_thresh": self.drop_score,
                "enable_mkldnn": self.enable_mkldnn,
            }
            if self.use_angle_cls:
                kwargs["textline_orientation_batch_size"] = 1
            return kwargs

        def _build_legacy_init_kwargs(self, lang_code: str, use_gpu: bool):
            model_root = os.path.join(PADDLE_OCR_PATH, lang_code, self.ocr_version)
            kwargs = {
                "lang": lang_code,
                "ocr_version": self.ocr_version,
                "use_gpu": use_gpu,
                "use_angle_cls": self.use_angle_cls,
                "enable_mkldnn": self.enable_mkldnn,
                "det_limit_side_len": self.det_limit_side_len,
                "rec_batch_num": self.rec_batch_num,
                "drop_score": self.drop_score,
                "det_model_dir": os.path.join(model_root, "det"),
                "rec_model_dir": os.path.join(model_root, "rec"),
            }
            if self.use_angle_cls:
                kwargs["cls_model_dir"] = os.path.join(model_root, "cls")
            return kwargs

        def _run_ocr_model(self, img: np.ndarray):
            if self._paddleocr_api_style == "new" and hasattr(self.model, "predict"):
                result = self.model.predict(img)
                if (
                    not isinstance(result, (list, tuple))
                    and hasattr(result, "__iter__")
                    and not hasattr(result, "json")
                ):
                    result = list(result)
                return result
            return self.model.ocr(img, det=True, rec=True, cls=self.use_angle_cls)

        def uses_internal_text_detector(self) -> bool:
            return self.ocr_version == "PP-OCRv5"

        def _result_items(self, result):
            if result is None:
                return []

            if hasattr(result, "json"):
                try:
                    data = result.json
                    if callable(data):
                        data = data()
                    return self._result_items(data)
                except Exception:
                    return []

            if isinstance(result, dict):
                if "res" in result:
                    return self._result_items(result["res"])
                if "ocrResults" in result:
                    items = []
                    for item in result["ocrResults"]:
                        items.extend(self._result_items(item))
                    return items
                return [result]

            if isinstance(result, (list, tuple)):
                items = []
                for item in result:
                    items.extend(self._result_items(item))
                return items

            return []

        def _get_indexed_result_value(self, values, index):
            if values is None:
                return None
            try:
                if len(values) <= index:
                    return None
                return values[index]
            except TypeError:
                return None

        def _poly_from_result(self, polys, boxes, index):
            poly = self._get_indexed_result_value(polys, index)
            if poly is not None:
                poly = np.asarray(poly, dtype=np.float32)
                if poly.size == 8:
                    return poly.reshape(4, 2)
                if poly.shape == (4, 2):
                    return poly

            box = self._get_indexed_result_value(boxes, index)
            if box is None:
                return None
            box = np.asarray(box, dtype=np.float32).reshape(-1)
            if box.size < 4:
                return None
            x1, y1, x2, y2 = box[:4]
            return np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )

        def _textblock_from_ocr_poly(self, poly: np.ndarray, text: str) -> TextBlock:
            poly = np.round(poly).astype(np.int64)
            x1, y1 = poly[:, 0].min(), poly[:, 1].min()
            x2, y2 = poly[:, 0].max(), poly[:, 1].max()
            width = max(
                float(np.linalg.norm(poly[1] - poly[0])),
                float(np.linalg.norm(poly[2] - poly[3])),
            )
            height = max(
                float(np.linalg.norm(poly[3] - poly[0])),
                float(np.linalg.norm(poly[2] - poly[1])),
            )
            src_is_vertical = height > width * 1.2
            font_size = max(height if not src_is_vertical else width, 1.0)
            language = "ja" if self.language == "Japanese" else "unknown"
            blk = TextBlock(
                xyxy=[int(x1), int(y1), int(x2), int(y2)],
                lines=[poly.tolist()],
                language=language,
                text=text,
                src_is_vertical=src_is_vertical,
                _detected_font_size=font_size,
                det_model="paddle_ocr",
            )
            blk.vertical = src_is_vertical
            return blk

        def _extract_textblocks(self, result) -> List[TextBlock]:
            blk_list = []
            for item in self._result_items(result):
                rec_texts = item.get("rec_texts")
                if rec_texts is None or len(rec_texts) == 0:
                    continue

                rec_scores = item.get("rec_scores")
                rec_polys = item.get("rec_polys")
                if rec_polys is None:
                    rec_polys = item.get("dt_polys")
                rec_boxes = item.get("rec_boxes")

                for idx, raw_text in enumerate(rec_texts):
                    if raw_text is None:
                        continue
                    score = self._get_indexed_result_value(rec_scores, idx)
                    if score is not None and float(score) < self.drop_score:
                        continue
                    poly = self._poly_from_result(rec_polys, rec_boxes, idx)
                    if poly is None:
                        continue
                    text = self._process_texts([str(raw_text)])
                    if not text:
                        continue
                    blk_list.append(self._textblock_from_ocr_poly(poly, text))
            return blk_list

        def _mask_from_textblocks(
            self, img_shape: tuple, blk_list: List[TextBlock]
        ) -> np.ndarray:
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            for blk in blk_list:
                for line in blk.lines:
                    poly = np.asarray(line, dtype=np.int32)
                    if poly.size == 8:
                        poly = poly.reshape(4, 2)
                    if poly.shape == (4, 2):
                        cv2.fillPoly(mask, [poly], 255)
            return mask

        def detect_and_ocr(self, img: np.ndarray):
            if not self.uses_internal_text_detector():
                return None, []
            result = self._run_ocr_model(img)
            blk_list = self._extract_textblocks(result)
            mask = self._mask_from_textblocks(img.shape, blk_list)
            return mask, blk_list

        def _extract_texts(self, result):
            if result is None:
                return []

            if hasattr(result, "json"):
                try:
                    data = result.json
                    if callable(data):
                        data = data()
                    return self._extract_texts(data)
                except Exception:
                    return []

            if isinstance(result, dict):
                rec_texts = result.get("rec_texts")
                if isinstance(rec_texts, list):
                    return [str(text) for text in rec_texts if text is not None]

                if "res" in result:
                    return self._extract_texts(result["res"])

                if "ocrResults" in result:
                    texts = []
                    for item in result["ocrResults"]:
                        texts.extend(self._extract_texts(item))
                    return texts

            if isinstance(result, (list, tuple)):
                if not result:
                    return []

                if all(isinstance(item, str) for item in result):
                    return [item for item in result if item]

                if any(hasattr(item, "json") for item in result if item is not None):
                    texts = []
                    for item in result:
                        texts.extend(self._extract_texts(item))
                    return texts

                legacy_texts = []
                for line in result:
                    if (
                        isinstance(line, list)
                        or isinstance(line, tuple)
                    ) and len(line) > 1:
                        recog = line[1]
                        if isinstance(recog, (list, tuple)) and len(recog) > 0:
                            legacy_texts.append(str(recog[0]))

                if legacy_texts:
                    return legacy_texts

                if len(result) == 1:
                    return self._extract_texts(result[0])

            return []

        def _load_model(self):
            if self.model is not None:
                return

            if not PADDLE_RUNTIME_AVAILABLE:
                raise RuntimeError(PADDLE_RUNTIME_INSTALL_HINT)

            lang_code = self.lang_map[self.language]
            use_gpu = True if self.device == "cuda" else False
            if self.debug_mode:
                self.logger.info(
                    f"Loading PaddleOCR model for language: {self.language} ({lang_code}), GPU: {use_gpu}"
                )

            new_kwargs = self._build_new_init_kwargs(lang_code, use_gpu)
            legacy_kwargs = self._build_legacy_init_kwargs(lang_code, use_gpu)
            try:
                self.model = PaddleOCR(**new_kwargs)
                self._paddleocr_api_style = "new"
            except TypeError as exc:
                # PaddleOCR 3.5.0 prefers the new pipeline-style keyword names.
                # Keep a legacy fallback so older wheels continue to work.
                if "unexpected keyword" not in str(exc).lower():
                    raise
                if self.debug_mode:
                    self.logger.debug(
                        "Falling back to legacy PaddleOCR init kwargs: %s", exc
                    )
                self.model = PaddleOCR(**legacy_kwargs)
                self._paddleocr_api_style = "legacy"

        def _close_paddle_resource(self, resource):
            if resource is not None:
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                        return True
                    except Exception as exc:
                        LOGGER.warning("Failed to close PaddleOCR resource: %s", exc)
            return False

        def unload_model(self, empty_cache=False):
            if self.model is not None:
                if not self._close_paddle_resource(self.model):
                    self._close_paddle_resource(getattr(self.model, "paddlex_pipeline", None))
            return super().unload_model(empty_cache=empty_cache)

        def ocr_img(self, img: np.ndarray) -> str:
            if self.debug_mode:
                self.logger.debug(f"Starting OCR for image size: {img.shape}")
            result = self._run_ocr_model(img)
            if self.debug_mode:
                self.logger.debug(f"OCR recognition result: {result}")
            text = self._process_result(result)
            return text

        def _ocr_blk_list(
            self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
        ):
            im_h, im_w = img.shape[:2]
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                    cropped_img = img[y1:y2, x1:x2]
                    try:
                        result = self._run_ocr_model(cropped_img)
                        raw_texts = self._extract_texts(result)
                        raw_text = " ".join(raw_texts)

                        if self.debug_mode:
                            self.logger.debug(
                                f"Raw OCR text from block ({x1}, {y1}, {x2}, {y2}): {raw_text}"
                            )

                        # Process the OCR result
                        text = self._process_result(result)

                        if self.debug_mode:
                            self.logger.debug(
                                f"Processed text from block ({x1}, {y1}, {x2}, {y2}): {text}"
                            )

                        blk.text = text if text else ""

                    except Exception as e:
                        if self.debug_mode:
                            self.logger.error(f"Error recognizing block: {str(e)}")
                        blk.text = ""
                else:
                    if self.debug_mode:
                        self.logger.warning(
                            "Invalid text block coordinates for target image"
                        )
                    blk.text = ""

        def _process_texts(self, raw_texts: List[str]) -> str:
            if not raw_texts:
                return ""

            raw_texts = [text for text in raw_texts if text]
            if not raw_texts:
                return ""

            # Depending on the output_format, we concatenate the lines
            if self.output_format == "Single Line":
                joined_text = " ".join(raw_texts)
                # Text cleaning
                joined_text = re.sub(r"-(?!\w)", "", joined_text)
                joined_text = re.sub(r"\s+", " ", joined_text)
            elif self.output_format == "As Recognized":
                joined_text = " ".join(raw_texts)
                # Clean up text, preserve line breaks
                joined_text = re.sub(r"-(?!\w)", "", joined_text)
                joined_text = re.sub(r"\s+", " ", joined_text)
            else:
                joined_text = " ".join(raw_texts)
                joined_text = re.sub(r"-(?!\w)", "", joined_text)
                joined_text = re.sub(r"\s+", " ", joined_text)

            # Apply case conversion to all text
            processed_text = self._apply_text_case(joined_text)
            processed_text = self._apply_punctuation_and_spacing(processed_text)

            if self.debug_mode:
                self.logger.debug(f"Final processed text: {processed_text}")

            return processed_text

        def _process_result(self, result):
            try:
                raw_texts = self._extract_texts(result)
                return self._process_texts(raw_texts)
            except Exception as e:
                if self.debug_mode:
                    self.logger.error(f"Error processing OCR result: {str(e)}")
                return ""

        def _apply_text_case(self, text: str) -> str:
            if self.text_case == "Uppercase":
                return text.upper()
            elif self.text_case == "Capitalize Sentences":
                return self._capitalize_sentences(text)
            elif self.text_case == "Lowercase":
                return text.lower()
            else:
                return text  # No change if the mode is not recognized

        def _capitalize_sentences(self, text: str) -> str:
            def process_sentence(sentence):
                words = sentence.split()
                if not words:
                    return ""
                if len(words) == 1:
                    return words[0].capitalize()
                else:
                    return " ".join(
                        [words[0].capitalize()] + [word.lower() for word in words[1:]]
                    )

            # We divide into sentences only by punctuation marks
            sentences = re.split(r"(?<=[.!?…])\s+", text)
            return " ".join(process_sentence(sentence) for sentence in sentences)

        def _apply_punctuation_and_spacing(self, text: str) -> str:
            text = re.sub(r"\s+([,.!?…])", r"\1", text)
            text = re.sub(r"([,.!?…])(?!\s)(?![,.!?…])", r"\1 ", text)
            text = re.sub(r"([,.!?…])\s+([,.!?…])", r"\1\2", text)
            return text.strip()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in [
                "language",
                "device",
                "use_angle_cls",
                "ocr_version",
                "enable_mkldnn",
                "det_limit_side_len",
                "rec_batch_num",
                "drop_score",
            ]:
                self.language = self.params["language"]["value"]
                self.device = self.params["device"]["value"]
                self.use_angle_cls = self.params["use_angle_cls"]["value"]
                self.ocr_version = self.params["ocr_version"]["value"]
                self.enable_mkldnn = self.params["enable_mkldnn"]["value"]
                self.det_limit_side_len = self.params["det_limit_side_len"]["value"]
                self.rec_batch_num = self.params["rec_batch_num"]["value"]
                self.drop_score = self.params["drop_score"]["value"]
                self.unload_model(empty_cache=True)
                self._load_model()
            elif param_key == "text_case":
                self.text_case = self.params["text_case"]["value"]
            elif param_key == "output_format":
                self.output_format = self.params["output_format"]["value"]

else:
    # If PaddleOCR is not installed, you can define a stub or alternative module
    logging.info("PaddleOCR module will not be loaded as the library is not installed.")
