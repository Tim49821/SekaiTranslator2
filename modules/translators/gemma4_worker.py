import gc
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

try:
    from llama_cpp import Llama
except Exception as exc:
    Llama = None
    LLAMA_CPP_IMPORT_ERROR = exc
else:
    LLAMA_CPP_IMPORT_ERROR = None


def _build_messages(
    source_lang: str,
    target_lang: str,
    current_source: str,
    previous_source: Optional[str],
    previous_translation: Optional[str],
    thinking_mode: bool,
) -> List[Dict[str, str]]:
    previous_source = previous_source or "(none)"
    previous_translation = previous_translation or "(none)"
    thinking_instruction = (
        "Thinking mode is enabled, but the final answer must still contain only the current text cell translation."
        if thinking_mode
        else "Thinking mode is disabled. Do not output hidden reasoning, analysis, or thinking blocks."
    )
    system_prompt = (
        "You are a professional manga and comic translator. Translate only the current text cell. "
        "Preserve the speaker's tone, style, speech level, terminology, and consistency with the immediately previous text cell. "
        f"{thinking_instruction} "
        "Do not add explanations, alternatives, markdown, quotes, or labels. Output only the translated current text."
    )
    user_prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"Previous source text:\n{previous_source}\n\n"
        f"Previous translation:\n{previous_translation}\n\n"
        f"Current source text:\n{current_source}\n\n"
        "Translate the current source text only. Keep names, terminology, tone, style, and speech level consistent with the previous context."
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


def _empty_cache():
    gc.collect()


def _translate_one(llm, payload: Dict, source_text: str, previous_source: Optional[str], previous_translation: Optional[str]) -> str:
    messages = _build_messages(
        payload["source_lang"],
        payload["target_lang"],
        source_text,
        previous_source,
        previous_translation,
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
        return _clean_translation(_extract_content(response))
    finally:
        del messages
        if "response" in locals():
            del response
        _empty_cache()


def translate(payload: Dict) -> List[str]:
    llm = _load_model(payload)
    translations = []
    previous_source = None
    previous_translation = None

    for source_text in payload["texts"]:
        if not isinstance(source_text, str) or not source_text.strip():
            translations.append("")
            continue

        translated = _translate_one(
            llm,
            payload,
            source_text,
            previous_source,
            previous_translation,
        )
        translations.append(translated)
        previous_source = source_text
        previous_translation = translated

    del llm
    _empty_cache()
    return translations


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
