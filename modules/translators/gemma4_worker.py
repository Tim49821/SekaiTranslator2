import gc
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from llama_cpp import Llama
except Exception as exc:
    Llama = None
    LLAMA_CPP_IMPORT_ERROR = exc
else:
    LLAMA_CPP_IMPORT_ERROR = None


def _build_page_messages(
    source_lang: str,
    target_lang: str,
    indexed_texts: List[Tuple[int, str]],
    thinking_mode: bool,
) -> List[Dict[str, str]]:
    thinking_instruction = (
        "Thinking mode is enabled, but the final answer must still contain only the requested JSON translations."
        if thinking_mode
        else "Thinking mode is disabled. Do not output hidden reasoning, analysis, or thinking blocks."
    )
    system_prompt = (
        "You are a professional manga and comic translator. Translate every text cell from one page. "
        "Your highest priority is a natural, fluent translation that preserves the original writing style and keeps tone, speech level, characterization, terminology, and phrasing consistent across the whole page. "
        "Preserve each speaker's tone, style, speech level, terminology, and consistency across the whole page. "
        f"{thinking_instruction} "
        "Do not add explanations, alternatives, markdown, quotes, or labels. Output only valid JSON."
    )
    source_items = [
        {"id": idx + 1, "text": text}
        for idx, text in indexed_texts
    ]
    user_prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        "Translate the following page text cells in their original order. Treat all cells as shared page context.\n"
        "Prioritize natural wording, original style preservation, and consistent tone, speech level, terminology, and phrasing across all translated cells.\n"
        "Return a JSON array with the same ids and one translation per item, for example:\n"
        "[{\"id\":1,\"translation\":\"...\"},{\"id\":2,\"translation\":\"...\"}]\n\n"
        f"Page source texts:\n{json.dumps(source_items, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _load_model(payload: Dict):
    if Llama is None:
        raise RuntimeError(f"llama-cpp-python is not installed or failed to import: {LLAMA_CPP_IMPORT_ERROR}")

    model_path = payload["model_path"]
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Gemma4 GGUF model file not found: {model_path}")

    max_input_tokens = int(payload["max_input_tokens"])
    max_new_tokens = int(payload["max_new_tokens"])
    n_ctx = max(int(payload["context_tokens"]), max_input_tokens + max_new_tokens + 512)
    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": int(payload["gpu_layers"]),
        "verbose": False,
    }
    threads = int(payload.get("threads", 0))
    if threads > 0:
        kwargs["n_threads"] = threads
    return Llama(**kwargs)


def _extract_content(response) -> str:
    try:
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            text = choices[0].get("text")
            if isinstance(text, str):
                return text
    except Exception:
        pass
    return str(response)


def _clean_translation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip()
    text = text.strip("`")
    for prefix in (
        "Translation:",
        "Current translation:",
        "Translated text:",
        "Answer:",
        "Output:",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip().strip("\"'")


def _extract_json_array(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if isinstance(parsed, dict):
        for key in ("translations", "items", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    if isinstance(parsed, list):
        return parsed
    raise ValueError("Gemma4 page response did not contain a JSON array")


def _coerce_page_translations(response_text: str, expected_ids: List[int]) -> List[str]:
    items = _extract_json_array(response_text)

    if all(isinstance(item, str) for item in items):
        if len(items) != len(expected_ids):
            raise ValueError(f"Expected {len(expected_ids)} translations, got {len(items)}")
        return [_clean_translation(item) for item in items]

    by_id = {}
    ordered = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Gemma4 page response must contain strings or objects")
        item_id = item.get("id", item.get("index"))
        translation = item.get("translation", item.get("text", item.get("content")))
        if not isinstance(translation, str):
            raise ValueError("Gemma4 page response item is missing a translation string")
        if item_id is not None:
            try:
                by_id[int(item_id)] = _clean_translation(translation)
                continue
            except (TypeError, ValueError):
                pass
        ordered.append(_clean_translation(translation))

    if by_id:
        missing = [item_id for item_id in expected_ids if item_id not in by_id]
        if missing:
            raise ValueError(f"Gemma4 page response is missing ids: {missing}")
        return [by_id[item_id] for item_id in expected_ids]

    if len(ordered) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} translations, got {len(ordered)}")
    return ordered


def _empty_cache():
    gc.collect()


def _translate_page(llm, payload: Dict, indexed_texts: List[Tuple[int, str]]) -> List[str]:
    expected_ids = [idx + 1 for idx, _ in indexed_texts]
    messages = _build_page_messages(
        payload["source_lang"],
        payload["target_lang"],
        indexed_texts,
        bool(payload["thinking_mode"]),
    )
    temperature = float(payload["temperature"])
    kwargs = {
        "messages": messages,
        "max_tokens": int(payload["max_new_tokens"]),
        "temperature": temperature,
    }
    if temperature <= 0:
        kwargs["temperature"] = 0.0

    try:
        response = llm.create_chat_completion(**kwargs)
        return _coerce_page_translations(_extract_content(response), expected_ids)
    finally:
        del messages
        if "response" in locals():
            del response
        _empty_cache()


def translate(payload: Dict) -> List[str]:
    llm = _load_model(payload)
    try:
        indexed_texts = [
            (idx, source_text)
            for idx, source_text in enumerate(payload["texts"])
            if isinstance(source_text, str) and source_text.strip()
        ]
        translations = [""] * len(payload["texts"])
        if not indexed_texts:
            return translations

        page_translations = _translate_page(llm, payload, indexed_texts)
        for (source_idx, _), translated in zip(indexed_texts, page_translations):
            translations[source_idx] = translated
        return translations
    finally:
        del llm
        _empty_cache()


def main():
    try:
        payload = json.load(sys.stdin)
        translations = translate(payload)
        print(json.dumps({"translations": translations}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
