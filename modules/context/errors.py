"""Strict, provider-neutral classification of context-length failures."""

import json
import re
from collections.abc import Mapping
from numbers import Integral
from typing import Iterable, List, Optional, Tuple


class ContextLengthError(RuntimeError):
    """A provider rejected the request because its input context is too large."""


RECOGNIZED_STATUS_CODES = (400, 413, 422)
RECOGNIZED_CODES = (
    "context_length_exceeded",
    "context_window_exceeded",
    "input_too_long",
    "prompt_too_long",
    "too_many_input_tokens",
)
MESSAGE_PATTERNS = (
    r"\b(?:maximum|max)\s+context\s+(?:length|window)\b",
    r"\bcontext\s+(?:length|window)\b.{0,80}\b(?:exceed|overflow|too\s+long)",
    r"\b(?:exceed|overflow|too\s+long).{0,80}\bcontext\s+(?:length|window)\b",
    r"\b(?:maximum|max)\s+input\s+tokens?\b",
    r"\binput\s+tokens?\b.{0,80}\b(?:exceed|too\s+many)\b",
    r"\bprompt\s+(?:is\s+)?too\s+long\b",
)

_MESSAGE_REGEXES = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in MESSAGE_PATTERNS
)
_MISSING = object()
_CODE_KEYS = ("code", "type")
_STATUS_KEYS = ("status_code", "status", "http_status", "httpStatus")


def _member(value, name: str, default=None):
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except (AttributeError, TypeError):
        return default


def _json_payload(value):
    """Decode a response/body value when it exposes JSON."""

    if value is None:
        return None
    if isinstance(value, (Mapping, list, tuple)):
        return value
    json_member = _member(value, "json", _MISSING)
    if json_member is not _MISSING:
        try:
            payload = json_member() if callable(json_member) else json_member
        except Exception:
            payload = None
        if isinstance(payload, (Mapping, list, tuple)):
            return payload
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, (Mapping, list, tuple)) else None
    return None


def _response(error):
    response = _member(error, "response", None)
    return response


def _response_json(error):
    response = _response(error)
    return _json_payload(response)


def _exception_body(error):
    body = _member(error, "body", None)
    return _json_payload(body) or body


def _nested_mappings(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        yield value
        for child in value.values():
            for nested in _nested_mappings(child, seen):
                yield nested
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for child in value:
            for nested in _nested_mappings(child, seen):
                yield nested


def _message_values(value, nested_error_only=False) -> List[str]:
    values: List[str] = []
    for mapping in _nested_mappings(value):
        for key, child in mapping.items():
            key_text = str(key).casefold()
            if key_text == "message" and isinstance(child, str) and child.strip():
                values.append(child.strip())
            elif nested_error_only and key_text == "error":
                values.extend(_message_values(child, nested_error_only=False))
    return values


def _response_text(error) -> str:
    response = _response(error)
    text = _member(response, "text", "")
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return text.strip() if isinstance(text, str) else ""


def provider_error_message(error) -> str:
    """Extract a concise provider message with response JSON precedence."""

    response_json = _response_json(error)
    nested_messages = _message_values(response_json, nested_error_only=True)
    if nested_messages:
        return nested_messages[0]
    # Some providers return a top-level JSON message rather than an ``error``
    # object; it is still preferable to an opaque exception representation.
    top_level_messages = _message_values(response_json)
    if top_level_messages:
        return top_level_messages[0]

    response_text = _response_text(error)
    if response_text:
        return response_text

    body = _exception_body(error)
    body_messages = _message_values(body, nested_error_only=True)
    if body_messages:
        return body_messages[0]
    body_messages = _message_values(body)
    if body_messages:
        return body_messages[0]

    try:
        return str(error)
    except Exception:
        return ""


def _code_values(value) -> Iterable[str]:
    for mapping in _nested_mappings(value):
        for key, child in mapping.items():
            if str(key).casefold() not in _CODE_KEYS:
                continue
            if isinstance(child, str) and child.strip():
                yield child.strip()


def _direct_code_values(error) -> Iterable[str]:
    for name in _CODE_KEYS:
        value = _member(error, name, _MISSING)
        if isinstance(value, str) and value.strip():
            yield value.strip()


def _provider_error_codes(error) -> Tuple[str, ...]:
    values: List[str] = []
    roots = (
        error if isinstance(error, (Mapping, list, tuple)) else None,
        _exception_body(error),
        _response_json(error),
    )
    for value in _direct_code_values(error):
        if value.casefold() not in [item.casefold() for item in values]:
            values.append(value)
    for root in roots:
        for value in _code_values(root):
            if value.casefold() not in [item.casefold() for item in values]:
                values.append(value)
    # A response object can expose code/type as attributes without returning
    # them from json().  Include those fields after the exception itself.
    response = _response(error)
    for name in _CODE_KEYS:
        value = _member(response, name, _MISSING)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            if value.casefold() not in [item.casefold() for item in values]:
                values.append(value)
    return tuple(values)


def provider_error_code(error) -> str:
    """Return discovered provider code/type values in stable order."""

    return ", ".join(_provider_error_codes(error))


def _status_values(value) -> Iterable[int]:
    for mapping in _nested_mappings(value):
        for key, child in mapping.items():
            if str(key) not in _STATUS_KEYS:
                continue
            status = _coerce_status(child)
            if status is not None:
                yield status


def _coerce_status(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _known_statuses(error) -> Tuple[int, ...]:
    statuses: List[int] = []
    for name in _STATUS_KEYS:
        status = _coerce_status(_member(error, name, _MISSING))
        if status is not None:
            statuses.append(status)
    body = _exception_body(error)
    statuses.extend(_status_values(body))
    response_json = _response_json(error)
    statuses.extend(_status_values(response_json))
    response = _response(error)
    response_status = _coerce_status(_member(response, "status_code", _MISSING))
    if response_status is not None:
        statuses.append(response_status)
    return tuple(statuses)


def is_context_length_error(error) -> bool:
    """Classify only recognized context failures, never generic HTTP errors."""

    statuses = _known_statuses(error)
    if any(status not in RECOGNIZED_STATUS_CODES for status in statuses):
        return False

    recognized_codes = {code.casefold() for code in RECOGNIZED_CODES}
    if any(code.casefold() in recognized_codes for code in _provider_error_codes(error)):
        return True

    message = provider_error_message(error)
    return any(regex.search(message) for regex in _MESSAGE_REGEXES)
