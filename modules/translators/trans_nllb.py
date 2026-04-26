import gc
import os.path as osp
from copy import deepcopy
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseTranslator, register_translator
from ..base import DEVICE_SELECTOR, TORCH_DTYPE_MAP, soft_empty_cache


MODEL_PATH = "data/models/nllb-200-distilled-1.3B"


NLLB_LANG_MAP = {
    "简体中文": "zho_Hans",
    "繁體中文": "zho_Hant",
    "日本語": "jpn_Jpan",
    "English": "eng_Latn",
    "한국어": "kor_Hang",
    "Tiếng Việt": "vie_Latn",
    "čeština": "ces_Latn",
    "Nederlands": "nld_Latn",
    "Français": "fra_Latn",
    "Deutsch": "deu_Latn",
    "magyar nyelv": "hun_Latn",
    "Italiano": "ita_Latn",
    "Polski": "pol_Latn",
    "Português": "por_Latn",
    "Brazilian Portuguese": "por_Latn",
    "limba română": "ron_Latn",
    "русский язык": "rus_Cyrl",
    "Español": "spa_Latn",
    "Türk dili": "tur_Latn",
    "украї́нська мо́ва": "ukr_Cyrl",
    "Thai": "tha_Thai",
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva",
    "Malayalam": "mal_Mlym",
    "Tamil": "tam_Taml",
}


def _move_inputs_to_device(inputs, device: str):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


@register_translator("NLLB-200 distilled 1.3B")
class NLLB200DistilledTranslator(BaseTranslator):
    concate_text = False
    hf_model_repo_id = "facebook/nllb-200-distilled-1.3B"
    hf_model_save_dir = MODEL_PATH
    hf_model_required_files = [
        "config.json",
        ["tokenizer.json", "sentencepiece.bpe.model"],
        ["*.safetensors", "pytorch_model.bin"],
    ]
    hf_model_ignore_patterns = ["*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*"]

    params: Dict = {
        "description": (
            "Offline NLLB-200 distilled 1.3B translator. "
            "Place the Hugging Face snapshot in data/models/nllb-200-distilled-1.3B."
        ),
        "device": DEVICE_SELECTOR(),
        "precision": {
            "type": "selector",
            "options": ["auto", "fp32", "fp16", "bf16"],
            "value": "auto",
            "description": "Model loading dtype. Use auto to keep the checkpoint dtype.",
        },
        "low vram mode": {
            "type": "checkbox",
            "value": False,
            "description": "Unload the model after each translation call.",
        },
        "batch size": {
            "value": 4,
            "description": "Number of text cells translated together.",
        },
        "max input tokens": {
            "value": 512,
            "description": "Maximum source tokens per text cell.",
        },
        "max new tokens": {
            "value": 512,
            "description": "Maximum generated tokens per text cell.",
        },
    }

    _load_model_keys = {"model", "tokenizer"}

    def __init__(self, *args, **kwargs) -> None:
        self.params = deepcopy(type(self).params)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        super().__init__(*args, **kwargs)
        self.device = self.get_param_value("device")

    def _setup_translator(self):
        self.lang_map.update(NLLB_LANG_MAP)

    def _assert_model_dir(self):
        if not osp.isdir(MODEL_PATH):
            raise FileNotFoundError(
                f"NLLB-200 distilled 1.3B model directory not found: {MODEL_PATH}. "
                "Download facebook/nllb-200-distilled-1.3B and place the snapshot there."
            )

    def _dtype_kwargs(self) -> Dict:
        precision = self.get_param_value("precision")
        if precision == "auto":
            return {"dtype": "auto"}
        return {"dtype": TORCH_DTYPE_MAP[precision]}

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        self._assert_model_dir()
        self.device = self.get_param_value("device")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            **self._dtype_kwargs(),
        ).to(self.device).eval()

    def _translate_batch(self, src_batch: List[str], source_code: str, target_code: str) -> List[str]:
        self.tokenizer.src_lang = source_code
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_code)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise ValueError(f"Unsupported NLLB target language token: {target_code}")

        inputs = self.tokenizer(
            src_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.get_param_value("max input tokens")),
        )
        inputs = _move_inputs_to_device(inputs, self.device)

        with torch.inference_mode():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=int(self.get_param_value("max new tokens")),
            )

        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        return [text.strip() for text in translations]

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        if not self.all_model_loaded():
            self.load_model()

        source_code = self.lang_map[self.lang_source]
        target_code = self.lang_map[self.lang_target]
        batch_size = max(1, int(self.get_param_value("batch size")))
        translations = [""] * len(src_list)

        try:
            pending = [
                (idx, text)
                for idx, text in enumerate(src_list)
                if isinstance(text, str) and text.strip()
            ]
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                chunk_ids = [idx for idx, _ in chunk]
                chunk_texts = [text for _, text in chunk]
                chunk_translations = self._translate_batch(
                    chunk_texts,
                    source_code=source_code,
                    target_code=target_code,
                )
                for idx, translated in zip(chunk_ids, chunk_translations):
                    translations[idx] = translated
                del chunk, chunk_ids, chunk_texts, chunk_translations
                gc.collect()
                soft_empty_cache()
        finally:
            if self.low_vram_mode:
                self.unload_model(empty_cache=True)

        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "device":
            self.device = self.get_param_value("device")
        if param_key in {"device", "precision"}:
            self.unload_model(empty_cache=True)
