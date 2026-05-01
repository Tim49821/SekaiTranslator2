import gc
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from llama_cpp import Llama
except Exception as exc:
    Llama = None
    LLAMA_CPP_IMPORT_ERROR = exc
else:
    LLAMA_CPP_IMPORT_ERROR = None


CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
DEFAULT_STYLE_GUIDE = (
    "For Japanese-to-Korean manga translation, write 자연스러운 한국어 대사. "
    "Prefer fluent Korean dialogue over literal Japanese word order. Preserve character voice, "
    "honorifics, 반말/존댓말 shifts, emotional force, repeated catchphrases, and recurring terms. "
    "Keep speech-bubble wording compact. Render SFX and mimetic words naturally when they carry meaning; "
    "leave iconic sounds short when a Korean equivalent would feel forced. Correct only obvious OCR noise, "
    "especially broken punctuation, duplicated characters, or vertical text that was read in the wrong order."
)
GENERIC_STYLE_GUIDE = (
    "Write natural comic dialogue in the target language. Preserve character voice, speech level, "
    "terminology, emotional force, and page-level consistency. Keep speech-bubble wording compact and "
    "correct only obvious OCR noise when the intended text is clear from context."
)


def _payload_int(payload: Dict, key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _payload_float(payload: Dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _default_style_guide(source_lang: str, target_lang: str) -> str:
    if source_lang == "Japanese" and target_lang == "Korean":
        return DEFAULT_STYLE_GUIDE
    return GENERIC_STYLE_GUIDE


def _build_page_messages(
    source_lang: str,
    target_lang: str,
    indexed_texts: List[Tuple[int, str]],
    thinking_mode: bool,
    style_guide: str = "",
    context_texts: Optional[List[Tuple[int, str]]] = None,
    strict: bool = False,
) -> List[Dict[str, str]]:
    thinking_instruction = (
        "Thinking mode is enabled, but the final answer must still contain only the requested JSON translations."
        if thinking_mode
        else "Thinking mode is disabled. Do not output hidden reasoning, analysis, or thinking blocks."
    )
    style_instruction = style_guide.strip() or _default_style_guide(source_lang, target_lang)
    retry_instruction = (
        "This is a strict retry because the previous answer had invalid JSON, missing ids, or mismatched counts. "
        "Prioritize exact schema compliance over creativity. "
        if strict
        else ""
    )
    system_prompt = (
        "You are a professional manga and comic translator. Translate every requested text cell from one page. "
        "Your highest priority is natural, fluent dialogue that preserves the original writing style, tone, "
        "speech level, characterization, terminology, and phrasing consistency across the page. "
        f"{style_instruction} "
        f"{thinking_instruction} "
        f"{retry_instruction}"
        "Do not add explanations, alternatives, markdown, quotes, labels, or untranslated source repeats. "
        "Output only valid JSON."
    )
    source_items = [
        {"id": idx + 1, "text": text}
        for idx, text in indexed_texts
    ]
    context_items = [
        {"id": idx + 1, "text": text}
        for idx, text in (context_texts or [])
    ]
    context_prompt = ""
    if context_items:
        context_prompt = (
            "Nearby page context only. Use this for voice, speaker continuity, and terminology, "
            "but do not translate these context-only cells and do not include their ids in the output:\n"
            f"{json.dumps(context_items, ensure_ascii=False)}\n\n"
        )
    user_prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        "Translate the requested page text cells in their original order. Treat all cells as shared page context.\n"
        "Prioritize 자연스러운 한국어, 말투/존댓말 consistency, compact 말풍선 길이, terminology consistency, "
        "SFX handling, and OCR 잡음 보정 when the mistake is obvious from context.\n"
        "Return a JSON object with exactly one translation for each requested id, for example:\n"
        "{\"translations\":[{\"id\":1,\"translation\":\"...\"},{\"id\":2,\"translation\":\"...\"}]}\n\n"
        f"{context_prompt}"
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
        model_log_name = payload.get("model_log_name", "Gemma4 GGUF")
        raise FileNotFoundError(f"{model_log_name} model file not found: {model_path}")

    max_input_tokens = _payload_int(payload, "max_input_tokens", 4096)
    max_new_tokens = _payload_int(payload, "max_new_tokens", 2048)
    n_ctx = max(_payload_int(payload, "context_tokens", 8192), max_input_tokens + max_new_tokens + 512)
    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": int(payload["gpu_layers"]),
        "verbose": False,
    }
    threads = _payload_int(payload, "threads", 0)
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
    text = THINK_PATTERN.sub("", text)
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
    text = THINK_PATTERN.sub("", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start >= 0 and obj_end > obj_start:
            try:
                parsed = json.loads(text[obj_start : obj_end + 1])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
        if parsed is None:
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
        translation = item.get("translation", item.get("target", item.get("text", item.get("content"))))
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


def _message_token_count(llm, messages: List[Dict[str, str]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    tokenizer = getattr(llm, "tokenize", None)
    if callable(tokenizer):
        try:
            return len(tokenizer(serialized.encode("utf-8"), add_bos=False))
        except TypeError:
            try:
                return len(tokenizer(serialized.encode("utf-8")))
            except Exception:
                pass
        except Exception:
            pass
    return max(1, len(serialized) // 4)


def _context_for_chunk(
    indexed_texts: List[Tuple[int, str]],
    chunk: List[Tuple[int, str]],
    context_cells: int,
) -> List[Tuple[int, str]]:
    if context_cells <= 0 or not chunk:
        return []
    positions = {idx: pos for pos, (idx, _) in enumerate(indexed_texts)}
    start = positions[chunk[0][0]]
    end = positions[chunk[-1][0]]
    before = indexed_texts[max(0, start - context_cells):start]
    after = indexed_texts[end + 1:end + 1 + context_cells]
    return before + after


def _fit_context_to_budget(
    llm,
    payload: Dict,
    chunk: List[Tuple[int, str]],
    context_texts: List[Tuple[int, str]],
    strict: bool = False,
) -> List[Tuple[int, str]]:
    budget = max(128, _payload_int(payload, "max_input_tokens", 4096))
    context_texts = list(context_texts)
    while context_texts:
        messages = _build_page_messages(
            payload["source_lang"],
            payload["target_lang"],
            chunk,
            bool(payload["thinking_mode"]),
            payload.get("style_guide", ""),
            context_texts,
            strict=strict,
        )
        if _message_token_count(llm, messages) <= budget:
            break
        context_texts.pop()
    return context_texts


def _build_target_chunks(llm, payload: Dict, indexed_texts: List[Tuple[int, str]]) -> List[List[Tuple[int, str]]]:
    budget = max(128, _payload_int(payload, "max_input_tokens", 4096))
    full_messages = _build_page_messages(
        payload["source_lang"],
        payload["target_lang"],
        indexed_texts,
        bool(payload["thinking_mode"]),
        payload.get("style_guide", ""),
    )
    if _message_token_count(llm, full_messages) <= budget:
        return [indexed_texts]

    chunks = []
    current = []
    for item in indexed_texts:
        candidate = current + [item]
        messages = _build_page_messages(
            payload["source_lang"],
            payload["target_lang"],
            candidate,
            bool(payload["thinking_mode"]),
            payload.get("style_guide", ""),
        )
        if current and _message_token_count(llm, messages) > budget:
            chunks.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _create_completion(
    llm,
    payload: Dict,
    indexed_texts: List[Tuple[int, str]],
    context_texts: Optional[List[Tuple[int, str]]] = None,
    strict: bool = False,
) -> List[str]:
    expected_ids = [idx + 1 for idx, _ in indexed_texts]
    messages = _build_page_messages(
        payload["source_lang"],
        payload["target_lang"],
        indexed_texts,
        bool(payload["thinking_mode"]),
        payload.get("style_guide", ""),
        context_texts,
        strict=strict,
    )
    temperature = 0.0 if strict else _payload_float(payload, "temperature", 0.0)
    kwargs = {
        "messages": messages,
        "max_tokens": _payload_int(payload, "max_new_tokens", 2048),
        "temperature": max(0.0, temperature),
    }

    try:
        response = llm.create_chat_completion(**kwargs)
        return _coerce_page_translations(_extract_content(response), expected_ids)
    finally:
        del messages
        if "response" in locals():
            del response
        _empty_cache()


def _translate_chunk(
    llm,
    payload: Dict,
    indexed_texts: List[Tuple[int, str]],
    context_texts: Optional[List[Tuple[int, str]]] = None,
) -> List[str]:
    if not indexed_texts:
        return []
    retry_count = max(0, _payload_int(payload, "structure_retry_count", 1))
    last_exc = None
    try:
        return _create_completion(llm, payload, indexed_texts, context_texts, strict=False)
    except Exception as exc:
        last_exc = exc

    for _ in range(retry_count):
        try:
            strict_context = _fit_context_to_budget(llm, payload, indexed_texts, context_texts or [], strict=True)
            return _create_completion(llm, payload, indexed_texts, strict_context, strict=True)
        except Exception as exc:
            last_exc = exc

    if len(indexed_texts) > 1:
        mid = len(indexed_texts) // 2
        context_cells = max(0, _payload_int(payload, "chunk_context_cells", 2))
        left = indexed_texts[:mid]
        right = indexed_texts[mid:]
        left_context = list(context_texts or []) + right[:context_cells]
        right_context = list(context_texts or []) + left[-context_cells:]
        return (
            _translate_chunk(llm, payload, left, _fit_context_to_budget(llm, payload, left, left_context))
            + _translate_chunk(llm, payload, right, _fit_context_to_budget(llm, payload, right, right_context))
        )

    err_name = type(last_exc).__name__ if last_exc is not None else "StructureError"
    return [f"[ERROR: {err_name}]"]


def _cjk_text(text: str) -> str:
    return "".join(CJK_PATTERN.findall(text or ""))


def _looks_like_json_or_markdown(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith(("```", "{", "[")):
        return True
    return bool(re.search(r"\"(?:translations?|id|text|content)\"\s*:", stripped))


def _is_suspicious_translation(source: str, translation: str) -> bool:
    source = source or ""
    translation = translation or ""
    if source.strip() and not translation.strip():
        return True
    if _looks_like_json_or_markdown(translation):
        return True
    if THINK_PATTERN.search(translation):
        return True
    source_cjk = _cjk_text(source)
    translation_cjk = _cjk_text(translation)
    if len(source_cjk) >= 4 and source_cjk == translation_cjk:
        return True
    if len(source_cjk) >= 8 and source_cjk in translation_cjk:
        return True
    return False


def _repair_suspicious_translations(
    llm,
    payload: Dict,
    indexed_texts: List[Tuple[int, str]],
    translations: List[str],
) -> List[str]:
    repaired = list(translations)
    context_cells = max(0, _payload_int(payload, "chunk_context_cells", 2))
    for pos, ((source_idx, source_text), translated) in enumerate(zip(indexed_texts, repaired)):
        if not _is_suspicious_translation(source_text, translated):
            continue
        context = indexed_texts[max(0, pos - context_cells):pos] + indexed_texts[pos + 1:pos + 1 + context_cells]
        context = _fit_context_to_budget(llm, payload, [(source_idx, source_text)], context, strict=True)
        try:
            retry = _create_completion(llm, payload, [(source_idx, source_text)], context, strict=True)[0]
        except Exception:
            continue
        if retry.strip() and not _is_suspicious_translation(source_text, retry):
            repaired[pos] = retry
    return repaired


def _translate_page(llm, payload: Dict, indexed_texts: List[Tuple[int, str]]) -> List[str]:
    context_cells = max(0, _payload_int(payload, "chunk_context_cells", 2))
    ordered_results: Dict[int, str] = {}
    chunks = _build_target_chunks(llm, payload, indexed_texts)
    for chunk in chunks:
        context = _context_for_chunk(indexed_texts, chunk, context_cells)
        context = _fit_context_to_budget(llm, payload, chunk, context)
        chunk_translations = _translate_chunk(llm, payload, chunk, context)
        for (source_idx, _), translated in zip(chunk, chunk_translations):
            ordered_results[source_idx] = translated

    translations = [ordered_results.get(source_idx, "") for source_idx, _ in indexed_texts]
    return _repair_suspicious_translations(llm, payload, indexed_texts, translations)


def translate(payload: Dict) -> List[str]:
    payload.setdefault("structure_retry_count", 1)
    payload.setdefault("chunk_context_cells", 2)
    payload.setdefault("style_guide", "")
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
