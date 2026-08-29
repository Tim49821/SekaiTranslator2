"""Provider-independent token estimation and usage summaries.

The optional ``tiktoken`` dependency is deliberately isolated here.  A
missing package, an unknown model, or an encoder failure all use the same
small deterministic estimator so callers can still make conservative budget
decisions.
"""

from collections.abc import Mapping
from functools import lru_cache
import math
from numbers import Number
from typing import Dict, List, Optional, Tuple


MESSAGE_TOKEN_OVERHEAD = 4


@lru_cache(maxsize=64)
def _cached_token_encoding_for_model(model: str):
    import tiktoken  # type: ignore

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return None


def _token_encoding_for_model(model: str):
    try:
        return _cached_token_encoding_for_model(model)
    except Exception:
        return None


def fallback_token_count(text: str) -> int:
    """Estimate tokens consistently without a provider tokenizer.

    Runs of ASCII characters are charged at four characters per token and
    every non-ASCII character is charged as one token.  This intentionally
    favors a stable, dependency-free estimate over language-specific
    accuracy.
    """

    total = 0
    ascii_run = 0
    for character in text:
        if ord(character) < 128:
            ascii_run += 1
            continue
        if ascii_run:
            total += (ascii_run + 3) // 4
            ascii_run = 0
        total += 1
    if ascii_run:
        total += (ascii_run + 3) // 4
    return total


def messages_token_count(messages: List[Dict], model: str) -> int:
    """Estimate the input tokens represented by chat messages."""

    encoding = _token_encoding_for_model(model)
    total = 0
    for message in messages:
        content = str(message.get("content", ""))
        if encoding is None:
            content_tokens = fallback_token_count(content)
        else:
            try:
                content_tokens = len(encoding.encode(content))
            except Exception:
                content_tokens = fallback_token_count(content)
        total += MESSAGE_TOKEN_OVERHEAD + content_tokens
    return total


_MISSING = object()


def _usage_member(usage, name: str, default=None):
    """Read one usage member from either a mapping or an SDK object."""

    if usage is None:
        return default
    if isinstance(usage, Mapping):
        return usage.get(name, default)
    try:
        return getattr(usage, name)
    except (AttributeError, TypeError):
        return default


def _usage_count(value):
    """Return a usable non-negative numeric count, or ``None``."""

    if isinstance(value, bool) or not isinstance(value, Number):
        return None
    if isinstance(value, complex) and not isinstance(value, float):
        return None
    try:
        if value < 0:
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    # Token counts are normally integers.  Normalizing integral floats keeps
    # summaries compact while retaining unusual but valid numeric values.
    try:
        if value == int(value):
            return int(value)
    except (OverflowError, TypeError, ValueError):
        pass
    return value


def _first_usage_count(usage, names) -> Optional[Number]:
    for name in names:
        value = _usage_count(_usage_member(usage, name, _MISSING))
        if value is not None:
            return value
    return None


def _usage_details(usage):
    details = []
    for name in (
        "prompt_tokens_details",
        "prompt_token_details",
        "input_tokens_details",
        "input_token_details",
        "cache_details",
    ):
        detail = _usage_member(usage, name, None)
        if detail is not None:
            details.append(detail)
    return details


def _first_count_from_sources(sources, names) -> Optional[Number]:
    for source in sources:
        count = _first_usage_count(source, names)
        if count is not None:
            return count
    return None


def format_token_usage(usage) -> str:
    """Format aggregate usage fields in a stable, privacy-safe order."""

    if usage is None:
        return ""

    details = _usage_details(usage)
    sources = [usage] + details
    prompt = _first_count_from_sources(
        sources,
        ("prompt_tokens", "input_tokens", "prompt", "input"),
    )
    completion = _first_count_from_sources(
        sources,
        ("completion_tokens", "output_tokens", "completion", "output"),
    )
    total = _first_count_from_sources(
        [usage],
        ("total_tokens", "total"),
    )
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    cache_hit = _first_count_from_sources(
        sources,
        (
            "cache_hit",
            "cache_hit_tokens",
            "cached_tokens",
            "cache_read_tokens",
            "cache_read_input_tokens",
        ),
    )
    cache_miss = _first_count_from_sources(
        sources,
        ("cache_miss", "cache_miss_tokens"),
    )
    cache_write = _first_count_from_sources(
        sources,
        (
            "cache_write",
            "cache_write_tokens",
            "cache_creation_tokens",
            "cache_creation_input_tokens",
        ),
    )

    fields: Tuple[Tuple[str, Optional[Number]], ...] = (
        ("prompt", prompt),
        ("completion", completion),
        ("total", total),
        ("cache_hit", cache_hit),
        ("cache_miss", cache_miss),
        ("cache_write", cache_write),
    )
    return ", ".join(
        "{}={}".format(name, value)
        for name, value in fields
        if value is not None
    )


def format_completion_token_usage(completion) -> str:
    """Format the usage member of a provider completion object."""

    usage = _usage_member(completion, "usage", None)
    if usage is None:
        return ""
    return format_token_usage(usage)
