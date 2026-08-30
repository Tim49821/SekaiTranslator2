# LLM Translation Context, History, Glossary, and Token Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-page LLM translation history, deterministic glossary constraints, token-budget diagnostics, and context-overflow recovery to Sekai's fixed remote LLM providers and local Gemma translator.

**Architecture:** Build a provider-agnostic, Qt-free `modules/context/` package and a reusable `LLMContextAdapterMixin`. Keep project text authoritative, pass immutable page snapshots through a keyword-only translation boundary, and let the remote API and Gemma worker adapters render their own message formats while preserving Sekai's API-key, provider, prompt-preset, headless, relay, and local-worker behavior.

**Tech Stack:** Python 3.8+, `unittest`, `dataclasses`, Qt via `qtpy`, OpenAI-compatible Chat Completions, Pydantic, optional `tiktoken`, and the existing `llama-cpp-python` Gemma worker subprocess.

**Spec:** `docs/superpowers/specs/2026-08-30-llm-translation-context-design.md`

## Global Constraints

- Execute this as the second PR in the migration series, based on the completed PR 1 branch or its merge commit. If PR 1 is still open, stack PR 2 on that branch and retarget after PR 1 merges; do not reimplement custom-module discovery here.
- Reference [`SangGuKim/BallonsTranslator@0ca4965`](https://github.com/SangGuKim/BallonsTranslator/tree/0ca496533763e6945e8280989b174be1802fbded), but adapt namespaces and integration points instead of replacing Sekai's translator stack.
- Support `LLM OpenAI`, `LLM Google`, `LLM Grok`, `LLM OpenRouter`, `LLM Studio`, and `Gemma 4 E4B-it` in this PR.
- Use per-translator settings named exactly `context mode`, `history token budget`, `glossary path`, and `glossary mode`.
- Defaults are `page`, `4096`, empty path, and `matching`; page mode plus an empty glossary must preserve the existing remote system/user message sequence.
- Keep current fixed-provider classes, key-tier rotation, `.env` precedence, endpoint defaults, model catalog, Gemini request rules, Pydantic parsing, prompt presets, and Gemma subprocess lifecycle.
- History is runtime-only. Do not persist prompts or history messages in project JSON.
- Whole pages are indivisible history units; rebuild and recovery use `HISTORY_LOW_WATER_RATIO = 0.60`.
- Never drop the current page or selected glossary to recover context length.
- Diagnostics must not include source text, translations, prompts, glossary contents, endpoints, or API keys.
- Support Python 3.8 syntax: use `typing.List`, `typing.Dict`, `typing.Tuple`, and `typing.Optional`; do not use PEP 604 unions or built-in generic annotations.
- Do not modify `config/textstyles/default.json` or the PR 1 custom-modules design/plan documents; they are outside PR 2 scope.
- Do not add global `ModuleConfig` context fields or alter `config/config.json`; the existing `translator_params` serializer owns persistence.

**Verified baseline before implementation:**

```bash
.venv/bin/python -m unittest \
  tests.test_llm_prompt_presets \
  tests.test_llm_env \
  tests.test_llm_api_key_pools \
  tests.test_local_translators \
  tests.test_python_compat \
  -v
```

Result on 2026-08-30: 63 tests passed. Treat any failure in this same suite after a task as a regression introduced by PR 2 until proven otherwise.

---

### Task 1: Qt-free glossary loader and deterministic renderer

**Files:**
- Create: `modules/context/__init__.py`
- Create: `modules/context/glossary.py`
- Create: `tests/test_translator_glossary.py`

**Interfaces:**
- Produces: `GlossaryError(ValueError)`.
- Produces: immutable `GlossaryEntry(source: str, translation: str, note: str = "")`.
- Produces: `normalize_glossary_path(path: Optional[PathValue]) -> str`.
- Produces: `load_glossary(path: Optional[PathValue]) -> Tuple[GlossaryEntry, ...]`.
- Produces: `select_glossary(entries, sources, mode) -> Tuple[GlossaryEntry, ...]`.
- Produces: `render_glossary(entries) -> str`.
- Produces constants: `GLOSSARY_MODE_MATCHING`, `GLOSSARY_MODE_ALL`, and `GLOSSARY_MODES`.

- [ ] **Step 1: Write failing glossary format and selection tests**

Create `tests/test_translator_glossary.py` with these concrete cases:

```python
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.context.glossary import (
    GLOSSARY_MODE_ALL,
    GLOSSARY_MODE_MATCHING,
    GlossaryEntry,
    GlossaryError,
    load_glossary,
    normalize_glossary_path,
    render_glossary,
    select_glossary,
)


class TranslatorGlossaryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_supported_formats_preserve_file_order(self):
        expected = (
            GlossaryEntry("勇者", "용사", "title"),
            GlossaryEntry("魔王", "마왕", ""),
        )
        cases = {
            "terms.json": (
                '[{"src":"勇者","dst":"용사","info":"title"},'
                '{"src":"魔王","dst":"마왕"}]'
            ),
            "terms.tsv": "勇者\t용사\ttitle\n魔王\t마왕\n",
            "terms.txt": "# comment\n勇者->용사 # title\n魔王->마왕\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self.assertEqual(load_glossary(self.write(name, content)), expected)

    def test_matching_is_casefolded_literal_and_all_keeps_every_entry(self):
        entries = (
            GlossaryEntry("Hero", "용사"),
            GlossaryEntry("Mage", "마법사"),
        )
        self.assertEqual(
            select_glossary(entries, ["THE HERO arrives"], GLOSSARY_MODE_MATCHING),
            (entries[0],),
        )
        self.assertEqual(select_glossary(entries, ["none"], GLOSSARY_MODE_ALL), entries)

    def test_rendering_is_compact_unicode_json(self):
        self.assertEqual(
            render_glossary((GlossaryEntry("勇者", "용사", "title"),)),
            '{"glossary":[{"source":"勇者","translation":"용사","note":"title"}]}',
        )
        self.assertEqual(render_glossary(()), "")
```

- [ ] **Step 2: Add failing path, cache, duplicate, and error-location tests**

Append tests that verify expansion, cache invalidation, exact duplicate collapse, and concise errors:

```python
    def test_normalized_paths_share_cache_and_reload_after_change(self):
        path = self.write("terms.json", '[{"src":"A","dst":"가"}]')
        with patch.dict(os.environ, {"TEST_GLOSSARY_FILE": str(path)}):
            first = load_glossary("$TEST_GLOSSARY_FILE")
            second = load_glossary(path)
        self.assertIs(first, second)
        self.assertEqual(normalize_glossary_path(path), str(path.resolve()))

        path.write_text('[{"src":"A","dst":"나"}]', encoding="utf-8")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        self.assertEqual(load_glossary(path)[0].translation, "나")

    def test_duplicate_rows_collapse_but_conflicting_targets_fail(self):
        duplicate = self.write("duplicate.tsv", "Hero\t용사\nHero\t용사\n")
        self.assertEqual(load_glossary(duplicate), (GlossaryEntry("Hero", "용사"),))

        conflict = self.write("conflict.tsv", "Hero\t용사\nhero\t영웅\n")
        with self.assertRaisesRegex(GlossaryError, "line 2"):
            load_glossary(conflict)

    def test_missing_unsupported_and_malformed_files_report_the_path(self):
        missing = self.root / "missing.json"
        with self.assertRaisesRegex(GlossaryError, "Glossary file not found"):
            load_glossary(missing)

        unsupported = self.write("terms.csv", "Hero,용사\n")
        with self.assertRaisesRegex(GlossaryError, "Unsupported glossary format"):
            load_glossary(unsupported)

        malformed = self.write("bad.json", "{")
        with self.assertRaisesRegex(GlossaryError, "line 1"):
            load_glossary(malformed)

    def test_invalid_mode_fails_concisely(self):
        with self.assertRaisesRegex(GlossaryError, "Invalid glossary mode"):
            select_glossary((GlossaryEntry("A", "가"),), ["A"], "unknown")
```

- [ ] **Step 3: Run glossary tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_translator_glossary -v`

Expected: ERROR because `modules.context.glossary` does not exist.

- [ ] **Step 4: Implement the full glossary API**

Adapt the GPL-compatible reference implementation into `modules/context/glossary.py`, retaining these exact public definitions and cache key:

```python
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import stat
from typing import Iterable, List, Optional, Sequence, Tuple, Union

GLOSSARY_MODE_MATCHING = "matching"
GLOSSARY_MODE_ALL = "all"
GLOSSARY_MODES = (GLOSSARY_MODE_MATCHING, GLOSSARY_MODE_ALL)
_COMMENT_PREFIXES = ("#", "//", "\\\\")
_TEXT_SUFFIXES = (".txt", ".tsv")
PathValue = Union[str, os.PathLike]


class GlossaryError(ValueError):
    pass


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    translation: str
    note: str = ""


def normalize_glossary_path(path: Optional[PathValue]) -> str:
    if path is None:
        return ""
    raw_path = os.fspath(path)
    if isinstance(raw_path, bytes):
        raw_path = os.fsdecode(raw_path)
    raw_path = raw_path.strip()
    if not raw_path:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    return os.path.realpath(os.path.abspath(os.path.normpath(expanded)))


def load_glossary(path: Optional[PathValue]) -> Tuple[GlossaryEntry, ...]:
    normalized_path = normalize_glossary_path(path)
    if not normalized_path:
        return ()
    try:
        file_stat = os.stat(normalized_path)
    except FileNotFoundError:
        raise GlossaryError('Glossary file not found: "{}".'.format(normalized_path)) from None
    except OSError as exc:
        raise GlossaryError('Could not access glossary "{}": {}.'.format(
            normalized_path, str(exc.strerror or exc).rstrip(".")
        )) from None
    if not stat.S_ISREG(file_stat.st_mode):
        raise GlossaryError('Glossary path is not a file: "{}".'.format(normalized_path))
    return _load_glossary_cached(
        normalized_path,
        file_stat.st_mtime_ns,
        file_stat.st_size,
    )


@lru_cache(maxsize=16)
def _load_glossary_cached(
    normalized_path: str,
    _mtime_ns: int,
    _size: int,
) -> Tuple[GlossaryEntry, ...]:
    text = Path(normalized_path).read_text(encoding="utf-8-sig")
    suffix = Path(normalized_path).suffix.casefold()
    if suffix == ".json":
        rows = _parse_json_rows(text, normalized_path)
    elif suffix in _TEXT_SUFFIXES:
        rows = _parse_text_rows(text, normalized_path, suffix)
    else:
        raise GlossaryError(
            'Unsupported glossary format for "{}"; expected .json, .txt, or .tsv.'.format(
                normalized_path
            )
        )
    return _deduplicate_rows(rows, normalized_path)
```

Implement `_parse_json_rows`, `_parse_text_rows`, `_make_entry`, `_deduplicate_rows`, `_line_error`, and `_location_error` with the exact formats and location rules in the spec. Implement selection and rendering exactly as follows:

```python
def select_glossary(
    entries: Iterable[GlossaryEntry],
    sources: Iterable[str],
    mode: str,
) -> Tuple[GlossaryEntry, ...]:
    ordered_entries = tuple(entries)
    if mode == GLOSSARY_MODE_ALL:
        return ordered_entries
    if mode != GLOSSARY_MODE_MATCHING:
        raise GlossaryError(
            'Invalid glossary mode "{}"; expected "matching" or "all".'.format(mode)
        )
    joined_sources = sources if isinstance(sources, str) else "\n".join(
        str(source) for source in sources if source is not None
    )
    folded_sources = joined_sources.casefold()
    return tuple(
        entry for entry in ordered_entries
        if entry.source.casefold() in folded_sources
    )


def render_glossary(entries: Sequence[GlossaryEntry]) -> str:
    if not entries:
        return ""
    payload = {
        "glossary": [
            {
                "source": entry.source,
                "translation": entry.translation,
                "note": entry.note,
            }
            for entry in entries
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
```

In `modules/context/__init__.py`, add only the package docstring at this stage:

```python
"""Shared LLM context, glossary, token-budget, and adapter helpers."""
```

- [ ] **Step 5: Run glossary tests to verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_translator_glossary -v`

Expected: PASS with every format, cache, selection, render, and error test green.

- [ ] **Step 6: Commit the glossary core**

```bash
git add modules/context/__init__.py modules/context/glossary.py tests/test_translator_glossary.py
git commit -m "feat: add LLM glossary core"
```

---

### Task 2: Token estimation, usage formatting, and context-length classification

**Files:**
- Create: `modules/context/token_usage.py`
- Create: `modules/context/errors.py`
- Create: `tests/test_llm_context_helpers.py`

**Interfaces:**
- Produces: `fallback_token_count(text: str) -> int`.
- Produces: `messages_token_count(messages: List[Dict], model: str) -> int`.
- Produces: `format_token_usage(usage) -> str` and `format_completion_token_usage(completion) -> str`.
- Produces: `ContextLengthError(RuntimeError)`.
- Produces: `provider_error_message(error) -> str`, `provider_error_code(error) -> str`, and `is_context_length_error(error) -> bool`.

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_llm_context_helpers.py`:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.context.errors import is_context_length_error, provider_error_message
from modules.context.token_usage import (
    fallback_token_count,
    format_completion_token_usage,
    format_token_usage,
    messages_token_count,
)


class LLMTokenUsageTest(unittest.TestCase):
    def test_fallback_is_deterministic_for_ascii_and_cjk(self):
        self.assertEqual(fallback_token_count("abcdefgh你"), 3)

    def test_unknown_model_uses_fallback_plus_message_overhead(self):
        with patch("modules.context.token_usage._token_encoding_for_model", return_value=None):
            self.assertEqual(
                messages_token_count([{"role": "user", "content": "abcdefgh你"}], "unknown"),
                7,
            )

    def test_usage_formats_openai_and_cache_fields(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 8},
        }
        self.assertEqual(
            format_token_usage(usage),
            "prompt=10, completion=2, total=12, cache_hit=8",
        )
        self.assertEqual(
            format_completion_token_usage(SimpleNamespace(usage=usage)),
            "prompt=10, completion=2, total=12, cache_hit=8",
        )


class LLMContextErrorTest(unittest.TestCase):
    def test_recognizes_context_codes_and_messages(self):
        coded = SimpleNamespace(
            code="context_length_exceeded",
            status_code=400,
            body={},
        )
        self.assertTrue(is_context_length_error(coded))
        self.assertTrue(is_context_length_error(RuntimeError("maximum context length exceeded")))

    def test_rejects_unrelated_bad_requests_and_non_input_statuses(self):
        self.assertFalse(is_context_length_error(RuntimeError("max_tokens is invalid")))
        self.assertFalse(is_context_length_error(SimpleNamespace(
            code="context_length_exceeded",
            status_code=500,
            body={},
        )))

    def test_nested_provider_message_is_extracted(self):
        response = SimpleNamespace(
            json=lambda: {"error": {"message": "prompt is too long"}},
            text="",
        )
        self.assertEqual(provider_error_message(SimpleNamespace(response=response)), "prompt is too long")
```

- [ ] **Step 2: Run helper tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_context_helpers -v`

Expected: ERROR because both helper modules are missing.

- [ ] **Step 3: Implement token estimation and usage normalization**

In `modules/context/token_usage.py`, implement an optional cached `tiktoken` lookup and the deterministic fallback:

```python
from collections.abc import Mapping
from functools import lru_cache
from typing import Dict, List

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
```

Add `_usage_member`, `_usage_count`, `format_token_usage`, and `format_completion_token_usage` using the exact normalized field order `prompt`, `completion`, `total`, `cache_hit`, `cache_miss`, `cache_write`. Accept mapping keys and object attributes, ignore booleans/negative/non-numeric values, and derive `total` only when both prompt and completion counts exist.

- [ ] **Step 4: Implement strict provider context-error classification**

In `modules/context/errors.py`, implement the reference extraction logic and use only these accepted HTTP statuses, codes, and message patterns:

```python
import re


class ContextLengthError(RuntimeError):
    pass


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
```

`provider_error_message()` must prefer nested response JSON error messages, then response text, then `str(error)`. `provider_error_code()` must collect top-level and nested `code`/`type` values from exception bodies and response JSON. `is_context_length_error()` must reject a known status outside `RECOGNIZED_STATUS_CODES` before testing codes or messages.

- [ ] **Step 5: Run helper tests to verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_llm_context_helpers -v`

Expected: PASS.

- [ ] **Step 6: Commit helper modules**

```bash
git add modules/context/token_usage.py modules/context/errors.py tests/test_llm_context_helpers.py
git commit -m "feat: add LLM context diagnostics helpers"
```

---

### Task 3: Immutable history window and whole-page budget operations

**Files:**
- Create: `modules/context/history.py`
- Modify: `tests/test_llm_context_helpers.py`

**Interfaces:**
- Produces enums: `ContextAction` and `ContextReason` with the values defined below.
- Produces immutable types: `HistoryPage`, `RenderedHistoryPage`, `HistoryWindowKey`, `HistoryWindow`, `ContextDiagnostic`, and `RequestContext`.
- Produces: `eligible_history_for_request(...) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]`.
- Produces: `window_rebuild_reason(...) -> Optional[ContextReason]`.
- Produces: `recover_context_length(request_context) -> Optional[RequestContext]`.

- [ ] **Step 1: Write failing rebuild, growth, eviction, and recovery tests**

Append to `tests/test_llm_context_helpers.py`:

```python
from collections import OrderedDict

from modules.context.history import (
    ContextAction,
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RenderedHistoryPage,
    RequestContext,
    eligible_history_for_request,
    recover_context_length,
    window_rebuild_reason,
)


def rendered(page_key, tokens):
    snapshot = HistoryPage(page_key, (page_key,), (page_key + "-translated",))
    return RenderedHistoryPage(snapshot, (("user", page_key), ("assistant", "done")), tokens)


class FakeProject:
    def __init__(self, keys):
        self.load_identity = object()
        self.pages = OrderedDict((key, []) for key in keys)


class LLMHistoryWindowTest(unittest.TestCase):
    def test_rebuild_selects_newest_suffix_below_low_water(self):
        project = FakeProject(["001.png", "002.png", "003.png", "004.png"])
        pages = {key: HistoryPage(key, (key,), ("done",)) for key in project.pages}
        history, diagnostic = eligible_history_for_request(
            window=None,
            project=project,
            page_key="004.png",
            previous_page=None,
            token_budget=10,
            rebuild_reason=ContextReason.WINDOW_EMPTY,
            snapshot_page=pages.get,
            render_page=lambda page: rendered(page.page_key, 3),
        )
        self.assertEqual([page.page_key for page in history], ["002.png", "003.png"])
        self.assertEqual(diagnostic.action, ContextAction.REBUILD)
        self.assertEqual(diagnostic.token_count, 6)

    def test_adjacent_request_grows_then_bulk_evicts(self):
        project = FakeProject(["001.png", "002.png", "003.png"])
        key = HistoryWindowKey(project.load_identity, (("model", "demo"),))
        first = rendered("001.png", 4)
        window = HistoryWindow(key, "001.png", (first,), 4)
        previous = HistoryPage("002.png", ("two",), ("둘",))

        history, diagnostic = eligible_history_for_request(
            window=window,
            project=project,
            page_key="003.png",
            previous_page=previous,
            token_budget=6,
            rebuild_reason=None,
            snapshot_page=lambda _key: None,
            render_page=lambda _page: rendered("002.png", 4),
        )
        self.assertEqual([page.page_key for page in history], ["002.png"])
        self.assertEqual(diagnostic.action, ContextAction.EVICT)
        self.assertEqual(diagnostic.evicted, 1)

    def test_context_recovery_removes_at_least_one_whole_page(self):
        request = RequestContext(
            history=(rendered("001.png", 3), rendered("002.png", 3)),
            history_budget=10,
            request_page_key="003.png",
        )
        recovered = recover_context_length(request)
        self.assertEqual([page.page_key for page in recovered.history], ["002.png"])
        self.assertEqual(recovered.diagnostic.action, ContextAction.CONTEXT_RECOVERY)

    def test_project_identity_and_nonadjacent_page_force_rebuild(self):
        project = FakeProject(["001.png", "002.png", "003.png"])
        key = HistoryWindowKey(project.load_identity, (("model", "demo"),))
        window = HistoryWindow(key, "001.png", (), 0)
        self.assertIsNone(window_rebuild_reason(window, project, "002.png", key))
        self.assertEqual(
            window_rebuild_reason(window, project, "003.png", key),
            ContextReason.NON_ADJACENT,
        )
```

- [ ] **Step 2: Run history tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_context_helpers.LLMHistoryWindowTest -v`

Expected: ERROR because `modules.context.history` is missing.

- [ ] **Step 3: Implement immutable state and diagnostics**

Create `modules/context/history.py` with `HISTORY_LOW_WATER_RATIO = 0.60` and these exact enum values and dataclass fields:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from .glossary import GlossaryEntry

HISTORY_LOW_WATER_RATIO = 0.60


class ContextAction(Enum):
    DISABLED = "disabled"
    EMPTY = "empty"
    REBUILD = "rebuild"
    REUSE = "reuse"
    GROW = "grow"
    EVICT = "evict"
    CONTEXT_RECOVERY = "context-recovery"


class ContextReason(Enum):
    HISTORY_DISABLED = "history-disabled"
    MISSING_PROJECT_PAGE = "missing-project-page"
    WINDOW_EMPTY = "window-empty"
    MISSING_LOAD_IDENTITY = "missing-load-identity"
    PROJECT_CHANGED = "project-changed"
    SETTINGS_CHANGED = "settings-changed"
    MISSING_PAGES = "missing-pages"
    NON_ADJACENT = "non-adjacent"
    SNAPSHOT_CHANGED = "snapshot-changed"
    PREVIOUS_INCOMPLETE = "previous-incomplete"
    OVERSIZED_PAGE = "oversized-page"


@dataclass(frozen=True)
class HistoryPage:
    page_key: str
    sources: Tuple[str, ...]
    translations: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedHistoryPage:
    snapshot: HistoryPage
    messages: Tuple[Tuple[str, str], ...]
    token_count: int

    @property
    def page_key(self) -> str:
        return self.snapshot.page_key


@dataclass(frozen=True)
class HistoryWindowKey:
    load_identity: object
    settings: Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class HistoryWindow:
    key: HistoryWindowKey
    request_page_key: str
    history: Tuple[RenderedHistoryPage, ...]
    token_count: int


@dataclass(frozen=True)
class ContextDiagnostic:
    page_key: str
    action: ContextAction
    page_count: int
    token_count: int
    token_budget: int
    appended: int = 0
    evicted: int = 0
    rebuild_reason: Optional[ContextReason] = None


@dataclass(frozen=True)
class RequestContext:
    history: Tuple[RenderedHistoryPage, ...]
    glossary: Tuple[GlossaryEntry, ...] = ()
    glossary_mode: str = ""
    history_budget: int = 0
    window_key: Optional[HistoryWindowKey] = None
    request_page_key: Optional[str] = None
    diagnostic: Optional[ContextDiagnostic] = None
```

Implement `ContextDiagnostic.__str__()` in the compact form `LLM Context: page=..., action=..., pages=..., tokens=.../...` and append only aggregate non-zero fields/reason.

- [ ] **Step 4: Implement rebuild, reuse, growth, eviction, and recovery**

Adapt the reference algorithms with these invariants:

```python
rebuild_limit = int(token_budget * HISTORY_LOW_WATER_RATIO)
low_water = int(token_budget * HISTORY_LOW_WATER_RATIO)
```

- Rebuild walks backward from the current page, skips ineligible snapshots, selects a newest suffix, and reverses it to chronological order.
- If the newest eligible page fits the hard budget but exceeds low water, keep that one page.
- Adjacent growth appends the complete preceding page.
- An oversized preceding page leaves the existing prefix untouched with reason `OVERSIZED_PAGE`.
- Overflow evicts whole oldest pages until both low-water and append constraints are satisfied.
- Recovery removes at least one whole oldest page and returns `None` when no history is removable.
- `window_rebuild_reason()` compares `load_identity` by identity (`is`), validates settings equality, and requires the current page to be directly after `window.request_page_key`.

- [ ] **Step 5: Run history and helper tests to verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_llm_context_helpers -v`

Expected: PASS.

- [ ] **Step 6: Commit immutable history support**

```bash
git add modules/context/history.py tests/test_llm_context_helpers.py
git commit -m "feat: add LLM history window budgeting"
```

---

### Task 4: Project translation identity and keyword-only translator boundary

**Files:**
- Modify: `utils/proj_imgtrans.py:91-222`
- Modify: `modules/translators/base.py:73-211`
- Create: `tests/test_llm_project_context.py`

**Interfaces:**
- Produces: `ProjImgTrans.load_identity -> object`.
- Produces: `ProjImgTrans.begin_full_page_translation(page_key: str) -> None`.
- Produces: `ProjImgTrans.mark_translation_finished(page_key: str, target_language: str) -> None`.
- Produces: `translation_is_successful(source: str, translation: str) -> bool`.
- Produces: `translation_request_covers_full_page(textblocks, project, page_key, full_page=False) -> bool`.
- Changes: `BaseTranslator.translate(..., *, project=None, page_key=None, commit_history_window=False)`.
- Produces hook: `BaseTranslator._translate_with_context(src_list, *, project=None, page_key=None, commit_history_window=False)`.
- Changes: `BaseTranslator.translate_textblk_lst(..., *, project=None, page_key=None, full_page=False) -> bool`.

- [ ] **Step 1: Write failing project identity and target metadata tests**

Create `tests/test_llm_project_context.py`:

```python
import unittest
from collections import OrderedDict
from unittest.mock import patch

from modules.translators.base import BaseTranslator
from utils.config import RunStatus
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import TextBlock


class ProjectTranslationContextTest(unittest.TestCase):
    def test_identity_changes_when_project_contents_are_reloaded(self):
        project = ProjImgTrans()
        project.directory = "/tmp/demo-project"
        first = project.load_identity
        with patch("utils.proj_imgtrans.find_all_imgs", return_value=[]):
            project.load_from_dict({"pages": {}, "image_info": {}})
        self.assertIsNot(project.load_identity, first)

    def test_begin_and_finish_manage_target_metadata(self):
        project = ProjImgTrans()
        project.pages = OrderedDict((
            ("001.png", [TextBlock(text=["hello"])]),
        ))
        project._image_info = {
            "001.png": {
                "finish_code": RunStatus.FIN_TRANSLATE,
                "translation_target": "English",
            }
        }

        project.begin_full_page_translation("001.png")
        self.assertFalse(project._image_info["001.png"]["finish_code"] & RunStatus.FIN_TRANSLATE)
        self.assertNotIn("translation_target", project._image_info["001.png"])

        project.mark_translation_finished("001.png", "한국어")
        self.assertTrue(project._image_info["001.png"]["finish_code"] & RunStatus.FIN_TRANSLATE)
        self.assertEqual(project._image_info["001.png"]["translation_target"], "한국어")
```

- [ ] **Step 2: Add failing translation-boundary and success tests**

Append:

```python
class RecordingTranslator(BaseTranslator):
    concate_text = False
    params = {}

    def _setup_translator(self):
        self.lang_map["日本語"] = "Japanese"
        self.lang_map["한국어"] = "Korean"
        self.context_call = None

    def _translate(self, src_list):
        return ["번역:" + text for text in src_list]

    def _translate_with_context(
        self,
        src_list,
        *,
        project=None,
        page_key=None,
        commit_history_window=False,
    ):
        self.context_call = (project, page_key, commit_history_window)
        return self._translate(src_list)


class TranslatorContextBoundaryTest(unittest.TestCase):
    def test_textblock_boundary_forwards_project_page_and_completion(self):
        translator = RecordingTranslator("日本語", "한국어")
        blocks = [TextBlock(text=["one"]), TextBlock(text=[""])]
        project = type("Project", (), {"pages": {"001.png": blocks}})()

        success = translator.translate_textblk_lst(
            blocks,
            project=project,
            page_key="001.png",
            full_page=True,
        )

        self.assertTrue(success)
        self.assertEqual(translator.context_call, (project, "001.png", True))
        self.assertEqual(blocks[0].translation, "번역:one")

    def test_error_markers_and_empty_outputs_are_not_successful(self):
        from modules.translators.base import translation_is_successful

        self.assertTrue(translation_is_successful("source", "target"))
        self.assertFalse(translation_is_successful("source", ""))
        self.assertFalse(translation_is_successful("source", "[ERROR: API Failed]"))
        self.assertTrue(translation_is_successful("", ""))
```

- [ ] **Step 3: Run boundary tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_project_context -v`

Expected: FAIL because the project identity/metadata methods and keyword-only boundary do not exist.

- [ ] **Step 4: Add project identity and translation-target lifecycle**

In `ProjImgTrans.__init__`, initialize `self._load_identity = object()`. Add:

```python
@property
def load_identity(self):
    return self._load_identity

def clear_page_progress(self, pagename: str, code: int):
    self._image_info[pagename]["finish_code"] &= ~code
    if code & RunStatus.FIN_TRANSLATE:
        self._image_info[pagename].pop("translation_target", None)

def begin_full_page_translation(self, page_key: str):
    self.clear_page_progress(page_key, RunStatus.FIN_TRANSLATE)

def mark_translation_finished(self, page_key: str, target_language: str):
    self.update_page_progress(page_key, RunStatus.FIN_TRANSLATE)
    self._image_info[page_key]["translation_target"] = target_language
```

Update `set_page_progress()` so clearing `FIN_TRANSLATE` removes the target. Replace `_load_identity` with a new object at the end of `load_from_dict()` and `new_project()`. Do not serialize the identity.

- [ ] **Step 5: Add the context-aware base hook without changing ordinary translators**

In `modules/translators/base.py`, add:

```python
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.proj_imgtrans import ProjImgTrans


def translation_is_successful(source: str, translation: str) -> bool:
    source = str(source or "")
    translation = str(translation or "")
    if not source.strip():
        return True
    if not translation.strip():
        return False
    return not translation.lstrip().startswith("[ERROR:")


def translation_request_covers_full_page(
    textblocks: List[TextBlock],
    project: Optional["ProjImgTrans"],
    page_key: Optional[str],
    full_page: bool = False,
) -> bool:
    if full_page:
        return True
    pages = getattr(project, "pages", None)
    page = pages.get(page_key) if isinstance(pages, dict) and page_key is not None else None
    if page is None:
        return False
    selected_ids = {id(block) for block in textblocks}
    return all(
        not block.get_text().strip() or id(block) in selected_ids
        for block in page
    )
```

Change `translate()` to accept the three keyword-only values and call this default hook:

```python
def _translate_with_context(
    self,
    src_list: List[str],
    *,
    project=None,
    page_key=None,
    commit_history_window: bool = False,
) -> List[str]:
    return self._translate(src_list)
```

In `translate_textblk_lst()`, capture `original_sources = [block.get_text() for block in textblk_lst]` before any hook runs, preserve preprocess/postprocess hooks, compute `commit_history_window = translation_request_covers_full_page(...)`, forward the keywords to `translate()`, and assign final translations. Return success with the same full-list alignment, including empty blocks:

```python
return all(
    translation_is_successful(source, translation)
    for source, translation in zip(original_sources, translations)
)
```

- [ ] **Step 6: Run boundary and existing translator tests**

Run: `.venv/bin/python -m unittest tests.test_llm_project_context tests.test_papago_translator tests.test_llm_api_key_pools -v`

Expected: PASS; existing ordinary translators ignore the new optional context values.

- [ ] **Step 7: Commit the project-aware boundary**

```bash
git add utils/proj_imgtrans.py modules/translators/base.py tests/test_llm_project_context.py
git commit -m "feat: add project-aware translation context boundary"
```

---

### Task 5: Shared context parameters and adapter mixin

**Files:**
- Create: `modules/context/params.py`
- Create: `modules/context/adapter.py`
- Create: `tests/test_llm_translation_context.py`

**Interfaces:**
- Produces: `build_llm_context_params() -> Dict` returning a fresh parameter dictionary.
- Produces: `CONTEXT_MODE_PAGE`, `CONTEXT_MODE_HISTORY`, `CONTEXT_MODES`, and `CONTEXT_INVALIDATION_KEYS`.
- Produces mixin: `LLMContextAdapterMixin`.
- Adapter requirements: `_context_model_name() -> str`, `_context_prompt_signature() -> str`, and `_render_history_page(page: HistoryPage) -> RenderedHistoryPage`.
- Adapter services: `_snapshot_request_context(project, page_key)`, `_commit_request_context(context)`, `_selected_glossary(context, sources)`, and `_clear_history_window()`.

- [ ] **Step 1: Write failing parameter isolation and adapter snapshot tests**

Create `tests/test_llm_translation_context.py`:

```python
import unittest
from collections import OrderedDict

from modules.context.adapter import LLMContextAdapterMixin
from modules.context.history import HistoryPage, RenderedHistoryPage
from modules.context.params import build_llm_context_params
from modules.context.token_usage import messages_token_count
from modules.translators.base import BaseTranslator
from utils.config import RunStatus
from utils.textblock import TextBlock


class FakeContextProject:
    def __init__(self):
        self.load_identity = object()
        first = TextBlock(text=["Hero"])
        first.translation = "용사"
        second = TextBlock(text=["Mage"])
        second.translation = "마법사"
        self.pages = OrderedDict((
            ("001.png", [first]),
            ("002.png", [second]),
            ("003.png", [TextBlock(text=["Current"])]),
        ))
        self._image_info = {
            "001.png": {"finish_code": RunStatus.FIN_TRANSLATE, "translation_target": "한국어"},
            "002.png": {"finish_code": RunStatus.FIN_TRANSLATE, "translation_target": "한국어"},
            "003.png": {"finish_code": 0},
        }


class FakeContextTranslator(LLMContextAdapterMixin, BaseTranslator):
    concate_text = False
    params = build_llm_context_params()

    def _setup_translator(self):
        self.lang_map["日本語"] = "Japanese"
        self.lang_map["한국어"] = "Korean"
        self._history_window = None

    def _translate(self, src_list, **_kwargs):
        return list(src_list)

    def _context_model_name(self):
        return "demo-model"

    def _context_prompt_signature(self):
        return "demo-prompt"

    def _render_history_page(self, page):
        messages = (
            ("user", "|".join(page.sources)),
            ("assistant", "|".join(page.translations)),
        )
        return RenderedHistoryPage(page, messages, 3)


class LLMContextAdapterTest(unittest.TestCase):
    def test_parameter_schema_is_fresh_and_uses_existing_file_picker(self):
        first = build_llm_context_params()
        second = build_llm_context_params()
        self.assertIsNot(first, second)
        self.assertEqual(first["context mode"]["options"], ["page", "history"])
        self.assertEqual(first["history token budget"]["value"], 4096)
        self.assertTrue(first["glossary path"]["editable"])
        self.assertTrue(first["glossary path"]["path_selector"])
        self.assertEqual(first["glossary mode"]["options"], ["matching", "all"])

    def test_history_snapshot_uses_only_complete_past_target_pages(self):
        translator = FakeContextTranslator(
            "日本語",
            "한국어",
            **{"context mode": "history", "history token budget": 10},
        )
        context = translator._snapshot_request_context(FakeContextProject(), "003.png")
        self.assertEqual([page.page_key for page in context.history], ["001.png", "002.png"])
        self.assertEqual(context.glossary, ())

    def test_page_mode_without_glossary_returns_none_and_clears_window(self):
        translator = FakeContextTranslator("日本語", "한국어")
        translator._history_window = object()
        self.assertIsNone(translator._snapshot_request_context(FakeContextProject(), "003.png"))
        self.assertIsNone(translator._history_window)
```

- [ ] **Step 2: Add failing invalidation, target, error-marker, and glossary tests**

Add cases that mutate a completed page to a different `translation_target`, an empty translation, and `[ERROR: API Failed]`, and assert `_snapshot_history_page()` rejects each. Add a temporary glossary file, set `glossary path`, assert `_snapshot_request_context()` freezes its entries, and assert `_selected_glossary()` returns matching terms only in matching mode. Also change `context mode`, `history token budget`, model/prompt inputs, and source/target language, then assert each change clears or rebuilds the committed window instead of reusing stale messages.

- [ ] **Step 3: Run adapter tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_translation_context.LLMContextAdapterTest -v`

Expected: ERROR because `params.py` and `adapter.py` are missing.

- [ ] **Step 4: Implement the shared parameter schema**

Create `modules/context/params.py`:

```python
from typing import Dict

CONTEXT_MODE_PAGE = "page"
CONTEXT_MODE_HISTORY = "history"
CONTEXT_MODES = (CONTEXT_MODE_PAGE, CONTEXT_MODE_HISTORY)
CONTEXT_INVALIDATION_KEYS = {
    "context mode",
    "history token budget",
    "glossary path",
    "glossary mode",
    "model",
    "override model",
    "system_prompt",
    "system prompt presets",
    "style guide",
    "style guide presets",
    "thinking mode",
}


def build_llm_context_params() -> Dict:
    return {
        "context mode": {
            "type": "selector",
            "options": list(CONTEXT_MODES),
            "value": CONTEXT_MODE_PAGE,
            "display_name": "LLM context",
            "description": "Use the current page alone or include eligible translated pages as history.",
        },
        "history token budget": {
            "value": 4096,
            "display_name": "History token budget",
            "description": "Estimated token budget reserved for complete prior-page examples.",
        },
        "glossary path": {
            "type": "selector",
            "options": [""],
            "value": "",
            "editable": True,
            "path_selector": True,
            "path_filter": "*.json *.txt *.tsv",
            "size": "median",
            "display_name": "Glossary file",
            "description": "Optional UTF-8 JSON, TXT, or TSV translation glossary.",
        },
        "glossary mode": {
            "type": "selector",
            "options": ["matching", "all"],
            "value": "matching",
            "display_name": "Glossary mode",
            "description": "Send matching current-page terms or the complete glossary.",
        },
    }
```

- [ ] **Step 5: Implement the adapter mixin**

Create `modules/context/adapter.py` with `LLMContextAdapterMixin`. It must normalize invalid parameter values to safe runtime values without rewriting unrelated settings, snapshot only immutable strings, and never retain `TextBlock` objects.

Use this exact window key shape:

```python
window_key = HistoryWindowKey(
    load_identity=getattr(project, "load_identity", None),
    settings=(
        ("source_language", str(self.lang_source)),
        ("target_language", str(self.lang_target)),
        ("model", self._context_model_name()),
        ("prompt_signature", self._context_prompt_signature()),
        ("token_budget", history_budget),
    ),
)
```

Use these exact lifecycle methods:

```python
def _clear_history_window(self):
    self._history_window = None

def _commit_request_context(self, request_context):
    if (
        request_context is None
        or request_context.window_key is None
        or request_context.request_page_key is None
    ):
        return
    self._history_window = HistoryWindow(
        key=request_context.window_key,
        request_page_key=request_context.request_page_key,
        history=request_context.history,
        token_count=sum(page.token_count for page in request_context.history),
    )

def _selected_glossary(self, request_context, sources):
    if request_context is None or not request_context.glossary:
        return ()
    if request_context.glossary_mode == GLOSSARY_MODE_ALL:
        return request_context.glossary
    return select_glossary(
        request_context.glossary,
        sources,
        request_context.glossary_mode,
    )
```

`_snapshot_history_page()` must require `FIN_TRANSLATE`, accept legacy missing target metadata, reject target mismatch, reject empty or `[ERROR:` translations, and return sources/translations aligned only to non-empty sources. `_snapshot_request_context()` must follow the rebuild/invalidation algorithm in the spec, log `ContextDiagnostic`, and return `None` only when both history and glossary are disabled.

Override `_translate_with_context()` in the mixin to freeze one `RequestContext` and call:

```python
return self._translate(
    src_list,
    request_context=request_context,
    page_key=page_key,
    commit_history_window=commit_history_window,
)
```

Provide default `updateParam()` and `unload_model()` implementations in the mixin, call `super()`, and clear the window for `CONTEXT_INVALIDATION_KEYS` or unload. This default applies only when the concrete translator does not define the same method; adapters with an existing override must explicitly invoke `_clear_history_window()` as described in their integration task.

- [ ] **Step 6: Run adapter, glossary, and history tests**

Run: `.venv/bin/python -m unittest tests.test_llm_translation_context.LLMContextAdapterTest tests.test_translator_glossary tests.test_llm_context_helpers -v`

Expected: PASS.

- [ ] **Step 7: Commit the shared adapter**

```bash
git add modules/context/params.py modules/context/adapter.py tests/test_llm_translation_context.py
git commit -m "feat: add shared LLM context adapter"
```

---

### Task 6: Remote fixed-provider history, glossary, usage, and overflow recovery

**Files:**
- Modify: `modules/translators/trans_llm_api_json.py:1-922`
- Modify: `tests/test_llm_translation_context.py`
- Modify: `tests/test_llm_prompt_presets.py`
- Modify: `tests/test_llm_api_key_pools.py`

**Interfaces:**
- Changes inheritance: `LLM_API_Translator(LLMContextAdapterMixin, BaseTranslator)`.
- Produces: `_render_user_prompt(queries, glossary_entries=()) -> str`.
- Produces: `_render_assistant_response(translations) -> str`.
- Produces: `_assemble_request(queries, request_context=None) -> Tuple[List[Dict], str]`.
- Changes: `_request_translation(messages, *, usage_page_key=None, usage_attempt=None) -> Optional[TranslationResponse]`.
- Changes: `_translate(src_list, *, request_context=None, page_key=None, commit_history_window=False) -> List[str]`.

- [ ] **Step 1: Write failing provider schema and compatibility-message tests**

Append to `tests/test_llm_translation_context.py`:

```python
from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    GrokLLMTranslator,
    LLMStudioTranslator,
    OpenAILLMTranslator,
    OpenRouterLLMTranslator,
)


class RemoteLLMContextTest(unittest.TestCase):
    provider_classes = (
        OpenAILLMTranslator,
        GoogleLLMTranslator,
        GrokLLMTranslator,
        OpenRouterLLMTranslator,
        LLMStudioTranslator,
    )

    def make_translator(self, **params):
        return OpenAILLMTranslator(
            "日本語",
            "한국어",
            raise_unsupported_lang=False,
            **{
                "free_api_keys": "test-key",
                "max requests per minute": 0,
                "delay": 0,
                **params,
            },
        )

    def test_every_fixed_provider_gets_independent_context_params(self):
        values = [provider.params["glossary path"] for provider in self.provider_classes]
        for provider in self.provider_classes:
            self.assertEqual(provider.params["context mode"]["value"], "page")
            self.assertEqual(provider.params["history token budget"]["value"], 4096)
        for index, value in enumerate(values):
            for other in values[index + 1:]:
                self.assertIsNot(value, other)

    def test_disabled_features_preserve_existing_two_message_shape(self):
        translator = self.make_translator()
        messages, prompt = translator._assemble_request(["こんにちは"])
        self.assertEqual(messages, [
            {"role": "system", "content": translator.system_prompt},
            {"role": "user", "content": prompt},
        ])
        self.assertIn("Please translate the following text snippets", prompt)
        self.assertIn('"id": 1', prompt)
```

- [ ] **Step 2: Add failing glossary placement and clean-history tests**

Add tests constructing `RequestContext` directly:

```python
    def test_matching_glossary_is_only_in_current_user_message(self):
        translator = self.make_translator()
        page = translator._render_history_page(
            HistoryPage("001.png", ("Hero",), ("용사",))
        )
        from modules.context.glossary import GlossaryEntry
        from modules.context.history import RequestContext
        context = RequestContext(
            history=(page,),
            glossary=(GlossaryEntry("Hero", "용사"), GlossaryEntry("Mage", "마법사")),
            glossary_mode="matching",
            history_budget=4096,
        )
        messages, _ = translator._assemble_request(["Mage appears"], context)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertNotIn("glossary", messages[1]["content"])
        self.assertIn('"source":"Mage"', messages[-1]["content"])
        self.assertNotIn('"source":"Hero"', messages[-1]["content"])

    def test_all_glossary_is_stable_system_message_before_history(self):
        translator = self.make_translator()
        from modules.context.glossary import GlossaryEntry
        from modules.context.history import RequestContext
        context = RequestContext(
            history=(),
            glossary=(GlossaryEntry("Hero", "용사"),),
            glossary_mode="all",
            history_budget=4096,
        )
        messages, _ = translator._assemble_request(["Nothing matches"], context)
        self.assertEqual([message["role"] for message in messages], ["system", "system", "user"])
        self.assertIn('"source":"Hero"', messages[1]["content"])
```

- [ ] **Step 3: Add failing immutable retry and context-recovery tests**

Use `unittest.mock.patch.object` to make `_request_translation()` raise one retryable connection error and then return a valid `TranslationResponse`; assert both calls receive equal messages. Add a separate test with two rendered history pages where `_request_translation()` raises `ContextLengthError` once and then succeeds; assert the retry has fewer complete history messages, the ordinary retry counter is untouched, and `_history_window` commits only after success. Add a final-failure test asserting the pre-existing committed window object remains unchanged.

- [ ] **Step 4: Run remote context tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_translation_context.RemoteLLMContextTest -v`

Expected: FAIL because fixed providers lack context params and remote requests accept only a prompt string.

- [ ] **Step 5: Attach the mixin and context parameters without disturbing provider copies**

Import `LLMContextAdapterMixin`, `build_llm_context_params`, glossary rendering, history types/recovery, token usage helpers, and context error helpers. Change the base class and merge a fresh schema into `LLM_API_Translator.params`:

```python
class LLM_API_Translator(LLMContextAdapterMixin, BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        # existing provider/key/model/prompt fields stay in their current order
        **build_llm_context_params(),
        # existing retry/sampling fields continue below
    }
```

Keep `_build_fixed_provider_params()` based on `deepcopy(LLM_API_Translator.params)` so every provider owns separate nested context metadata. Extend `LLM_TRANSLATOR_DEPENDENCIES` with `tiktoken>=0.7.0`; `token_usage.py` must still work through its fallback when the import is unavailable in unit tests.

Initialize `self._history_window = None` in `_setup_translator()`.

`LLM_API_Translator` already defines `updateParam()`, so that concrete method shadows the mixin implementation. Preserve all existing API-key-pool/client invalidation logic, then call `_clear_history_window()` when `param_key in CONTEXT_INVALIDATION_KEYS`. Add an assertion to the invalidation test proving a remote translator follows this path.

- [ ] **Step 6: Split current prompt rendering from message assembly**

Replace `_assemble_prompts()` with:

```python
def _render_user_prompt(self, queries, glossary_entries=()):
    from_lang = self.lang_map.get(self.lang_source, self.lang_source)
    to_lang = self.lang_map.get(self.lang_target, self.lang_target)
    input_elements = [
        {"id": index + 1, "source": query}
        for index, query in enumerate(queries)
    ]
    input_json = json.dumps(input_elements, ensure_ascii=False, indent=2)
    prompt = (
        f"Please translate the following text snippets from {from_lang} to {to_lang}. "
        "The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
        f"INPUT:\n{input_json}"
    )
    if glossary_entries:
        prompt += (
            "\n\nGLOSSARY:\nUse these mappings as wording constraints without changing "
            "the target language, ids, item count, or JSON format.\n"
            + render_glossary(glossary_entries)
        )
    return prompt


@staticmethod
def _render_assistant_response(translations):
    return json.dumps(
        {
            "translations": [
                {"id": index + 1, "translation": translation}
                for index, translation in enumerate(translations)
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
```

Implement `_render_history_page()` as a glossary-free user/assistant pair and count it with the active stripped model name. Implement `_context_model_name()` using `override_model or model` with the provider prefix removed; implement `_context_prompt_signature()` as `self.system_prompt`.

Implement `_assemble_request()` in this exact order: system prompt, optional all-glossary system constraint, chronological history pairs, current user prompt with matching glossary selection.

- [ ] **Step 7: Change the request boundary from prompt to messages and add usage logging**

Change `_request_translation()` to accept an already assembled `messages` list and pass it untouched into `api_args`. Around `client.chat.completions.create`, classify errors:

```python
try:
    completion = self.client.chat.completions.create(**api_args)
except Exception as exc:
    if is_context_length_error(exc):
        raise ContextLengthError(provider_error_message(exc)) from exc
    self.logger.error("API request failed: %s", type(exc).__name__)
    raise
```

After a completion, preserve `token_count` and `token_count_last` where available and log only the normalized aggregate:

```python
summary = format_completion_token_usage(completion)
if summary:
    self.logger.debug(
        "LLM token usage: page=%s, attempt=%s, %s",
        str(usage_page_key or "-").replace("\n", " ").replace("\r", " "),
        usage_attempt,
        summary,
    )
```

Keep response extraction, Pydantic validation, simple dictionary/list repair, ID ordering, and provider response-format code intact.

Replace the existing raw response-body debug log on validation failure with aggregate metadata such as response character count and exception type. Do not log `raw_content`, assembled messages, glossary JSON, endpoint values, or keys.

- [ ] **Step 8: Add whole-history recovery to the existing retry loop**

Change `_translate()` to accept `request_context`, `page_key`, and `commit_history_window`. Assemble messages once before ordinary retries. Catch `ContextLengthError` before the current retryable/fatal groups, call `recover_context_length(active_context)`, log its diagnostic, and rebuild only after successful whole-page eviction.

On success:

```python
if commit_history_window:
    self._commit_request_context(successful_context)
return translations
```

On any terminal marker result, do not call `_commit_request_context()`. When no history remains for a context-length error, return `[ERROR: ContextLengthError]` for each current source through the same terminal-result convention used by the existing translator.

- [ ] **Step 9: Run remote context and existing remote LLM regressions**

Run: `.venv/bin/python -m unittest tests.test_llm_translation_context.RemoteLLMContextTest tests.test_llm_prompt_presets tests.test_llm_api_key_pools tests.test_llm_env -v`

Expected: PASS; key rotation, provider prompt presets, environment precedence, and page-only request shape remain green.

- [ ] **Step 10: Commit remote adapter integration**

```bash
git add modules/translators/trans_llm_api_json.py tests/test_llm_translation_context.py tests/test_llm_prompt_presets.py tests/test_llm_api_key_pools.py
git commit -m "feat: add context to API LLM translators"
```

---

### Task 7: Gemma parent/worker history and glossary adapter

**Files:**
- Modify: `modules/translators/trans_gemma4.py:1-367`
- Modify: `modules/translators/gemma4_worker.py:1-521`
- Modify: `tests/test_local_translators.py:347-582`
- Modify: `tests/test_llm_translation_context.py`

**Interfaces:**
- Changes inheritance: `LocalGGUFTranslator(LLMContextAdapterMixin, BaseTranslator)`.
- Adds the shared four context parameters to `Gemma4E4BTranslator.params`.
- Adds worker helpers: `render_page_user_prompt(...) -> str`, `render_page_assistant_response(...) -> str`, and `_fit_history_pages_to_budget(...) -> List[Dict]`.
- Extends the worker payload with `history_pages`, `history_token_budget`, `glossary_json`, and `glossary_mode`.

- [ ] **Step 1: Write failing Gemma payload serialization test**

In `tests/test_local_translators.py`, extend `test_subprocess_runtime_calls_worker_with_gguf_payload` or add a focused test that builds a `RequestContext` with one rendered page and one glossary entry, calls `translator._translate(..., request_context=context)`, and asserts:

```python
self.assertEqual(payload["history_token_budget"], 4096)
self.assertEqual(payload["history_pages"][0]["page_key"], "001.png")
self.assertEqual(
    [message["role"] for message in payload["history_pages"][0]["messages"]],
    ["user", "assistant"],
)
self.assertIn('"source":"Hero"', payload["glossary_json"])
self.assertEqual(payload["glossary_mode"], "matching")
```

Also assert all context parameters exist in `Gemma4E4BTranslator.params` and are independent from `OpenAILLMTranslator.params`.

- [ ] **Step 2: Write failing worker ordering and exact-fit tests**

Add worker tests with `FakeLlama`:

```python
def test_worker_places_full_glossary_before_history_and_current_page(self):
    payload = self.base_worker_payload()
    payload.update({
        "history_pages": [
            {
                "page_key": "001.png",
                "messages": [
                    {"role": "user", "content": "previous source"},
                    {"role": "assistant", "content": "previous target"},
                ],
            }
        ],
        "history_token_budget": 4096,
        "glossary_json": '{"glossary":[{"source":"Hero","translation":"용사","note":""}]}',
        "glossary_mode": "all",
    })
    with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
         patch("modules.translators.gemma4_worker.Llama", FakeLlama):
        gemma4_worker.translate(payload)
    messages = FakeLlama.completion_calls[0]["messages"]
    self.assertEqual([message["role"] for message in messages], [
        "system", "system", "user", "assistant", "user"
    ])
    self.assertIn("glossary", messages[1]["content"])

def test_worker_drops_oldest_whole_history_pages_before_current_chunk(self):
    payload = self.base_worker_payload()
    payload["max_input_tokens"] = 120
    payload["history_token_budget"] = 40
    payload["history_pages"] = [
        {"page_key": "001.png", "messages": [
            {"role": "user", "content": "old " * 30},
            {"role": "assistant", "content": "old target " * 20},
        ]},
        {"page_key": "002.png", "messages": [
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent target"},
        ]},
    ]
    with patch("modules.translators.gemma4_worker.Path.is_file", return_value=True), \
         patch("modules.translators.gemma4_worker.Llama", FakeLlama):
        gemma4_worker.translate(payload)
    messages = FakeLlama.completion_calls[0]["messages"]
    serialized = "\n".join(message["content"] for message in messages)
    self.assertNotIn("old target", serialized)
    self.assertIn("recent target", serialized)
    self.assertIn("Page source texts", serialized)
```

Factor the repeated current worker payload into a concrete `base_worker_payload()` test helper containing every currently required key.

- [ ] **Step 3: Run Gemma context tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_local_translators.GemmaTranslatorTest -v`

Expected: FAIL because Gemma does not expose context params or serialize history/glossary.

- [ ] **Step 4: Attach the mixin and serialize immutable context**

Change the parent class to `LocalGGUFTranslator(LLMContextAdapterMixin, BaseTranslator)` and merge `build_llm_context_params()` into the Gemma parameter declaration. Keep the existing per-instance `deepcopy(type(self).params)`. Set `self._history_window = None` in `LocalGGUFTranslator._setup_translator()` after extending the language map.

Implement:

```python
def _context_model_name(self):
    return self.model_filename

def _context_prompt_signature(self):
    return json.dumps(
        {
            "thinking_mode": self.thinking_mode,
            "style_guide": str(self._optional_param_value("style guide", "")),
            "prompt_version": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def _render_history_page(self, page):
    messages = [
        {
            "role": "user",
            "content": gemma4_worker.render_page_user_prompt(
                self.lang_map[self.lang_source],
                self.lang_map[self.lang_target],
                list(enumerate(page.sources)),
            ),
        },
        {
            "role": "assistant",
            "content": gemma4_worker.render_page_assistant_response(page.translations),
        },
    ]
    return RenderedHistoryPage(
        page,
        tuple((message["role"], message["content"]) for message in messages),
        messages_token_count(messages, self.model_filename),
    )
```

Change `_translate()` to accept the context keywords. Render only the selected glossary entries and serialize pages as plain strings:

```python
history_pages = []
if request_context is not None:
    history_pages = [
        {
            "page_key": page.page_key,
            "messages": [
                {"role": role, "content": content}
                for role, content in page.messages
            ],
        }
        for page in request_context.history
    ]
selected_glossary = self._selected_glossary(request_context, src_list)
payload.update({
    "history_pages": history_pages,
    "history_token_budget": request_context.history_budget if request_context else 0,
    "glossary_json": render_glossary(selected_glossary),
    "glossary_mode": request_context.glossary_mode if request_context else "matching",
})
```

Commit the request context only when the worker returns a correctly sized list and every non-empty source has a non-empty non-error translation.

- [ ] **Step 5: Refactor worker prompt rendering into reusable functions**

Extract the current user prompt and assistant JSON into:

```python
def render_page_user_prompt(
    source_lang: str,
    target_lang: str,
    indexed_texts: List[Tuple[int, str]],
    context_texts: Optional[List[Tuple[int, str]]] = None,
    glossary_json: str = "",
) -> str:
    source_items = [{"id": idx + 1, "text": text} for idx, text in indexed_texts]
    context_items = [{"id": idx + 1, "text": text} for idx, text in (context_texts or [])]
    sections = [
        f"Source language: {source_lang}",
        f"Target language: {target_lang}",
        "Translate the requested page text cells in their original order. Treat all cells as shared page context.",
    ]
    if context_items:
        sections.append(
            "Nearby page context only. Do not translate these ids:\n"
            + json.dumps(context_items, ensure_ascii=False)
        )
    if glossary_json:
        sections.append(
            "Use these glossary mappings as wording constraints without changing ids or output format:\n"
            + glossary_json
        )
    sections.append("Page source texts:\n" + json.dumps(source_items, ensure_ascii=False))
    return "\n\n".join(sections)


def render_page_assistant_response(translations) -> str:
    return json.dumps(
        {
            "translations": [
                {"id": index + 1, "translation": translation}
                for index, translation in enumerate(translations)
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
```

Preserve the current Korean-specific quality instructions in `_build_page_messages()`; the helper extraction must not remove them from the system or current-page prompt.

- [ ] **Step 6: Add exact-token whole-page history fitting in the worker**

Implement `_fit_history_pages_to_budget(llm, payload, base_messages)`:

- validate every payload page as a dictionary with a list of `{role, content}` strings;
- count each page as an indivisible message group using `_message_token_count()`;
- walk from newest to oldest;
- cap history by `history_token_budget` and by the remaining `max_input_tokens` after system/full-glossary/current messages;
- reverse selected pages back to chronological order;
- return a flattened list of message dictionaries.

Extend `_build_page_messages()` with explicit `history_messages`, `glossary_json`, and `glossary_mode` keyword arguments. It must order messages as primary system, optional all-glossary system, fitted history pairs, and current user. In matching mode, `glossary_json` belongs only in the current user prompt.

For every normal, strict-retry, recursive-split, and suspicious-repair completion:

1. Build the current system/glossary/current-user messages with no history.
2. Fit nearby current-page context and chunk boundaries against that no-history base, so prior pages never displace requested cells or their glossary.
3. Pass that exact base into `_fit_history_pages_to_budget()`.
4. Insert only the returned whole-page history messages before the current user message.

Update `_build_target_chunks()`, `_fit_context_to_budget()`, and `_create_completion()` to pass the same glossary arguments on every path. Add assertions that strict retry and suspicious repair retain glossary placement and use the same newest history suffix that still fits their exact base-message budget.

- [ ] **Step 7: Run Gemma and shared context tests**

Run: `.venv/bin/python -m unittest tests.test_local_translators tests.test_llm_translation_context -v`

Expected: PASS; existing full-page generation, chunk splitting, strict retry, suspicious repair, quantization, and worker-runtime tests remain green.

- [ ] **Step 8: Commit Gemma adapter integration**

```bash
git add modules/translators/trans_gemma4.py modules/translators/gemma4_worker.py tests/test_local_translators.py tests/test_llm_translation_context.py
git commit -m "feat: add context to Gemma translation"
```

---

### Task 8: Wire project/page context through every translation pipeline

**Files:**
- Modify: `ui/module_manager.py:358-497,589-603,784-832,1520-1529,1614-1643`
- Modify: `ui/mainwindow.py:1543-1579,1735-1737`
- Create: `tests/test_llm_context_pipeline.py`

**Interfaces:**
- Produces module helper: `translate_project_textblocks(translator, project, page_key, blocks, full_page=False) -> bool`.
- Changes: `TranslateThread._translate_page(project, page_key, emit_finished=True) -> bool`.
- Changes: `TranslateThread.translatePage(project, page_key) -> None`.
- Changes block pipeline methods to carry `page_key` and the shared `ProjImgTrans` instance.

- [ ] **Step 1: Write failing pure pipeline-boundary tests**

Create `tests/test_llm_context_pipeline.py` and set `QT_QPA_PLATFORM=offscreen` before Qt imports. Test the module-level helper with fakes:

```python
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.module_manager import translate_project_textblocks
from utils.textblock import TextBlock


class FakeProject:
    def __init__(self, blocks):
        self.pages = {"001.png": blocks}
        self.events = []

    def begin_full_page_translation(self, page_key):
        self.events.append(("begin", page_key))

    def mark_translation_finished(self, page_key, target):
        self.events.append(("finish", page_key, target))


class FakeTranslator:
    lang_target = "한국어"

    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def translate_textblk_lst(self, blocks, **kwargs):
        self.calls.append((blocks, kwargs))
        if self.success:
            for block in blocks:
                if block.get_text().strip():
                    block.translation = "번역"
        return self.success


class LLMContextPipelineTest(unittest.TestCase):
    def test_full_page_invalidates_then_marks_only_after_success(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject(blocks)
        translator = FakeTranslator()
        self.assertTrue(translate_project_textblocks(
            translator, project, "001.png", blocks, full_page=True
        ))
        self.assertEqual(project.events, [
            ("begin", "001.png"),
            ("finish", "001.png", "한국어"),
        ])
        self.assertEqual(translator.calls[0][1], {
            "project": project,
            "page_key": "001.png",
            "full_page": True,
        })

    def test_failed_full_page_stays_invalidated(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject(blocks)
        translator = FakeTranslator(success=False)
        self.assertFalse(translate_project_textblocks(
            translator, project, "001.png", blocks, full_page=True
        ))
        self.assertEqual(project.events, [("begin", "001.png")])

    def test_partial_selection_uses_context_without_completing_page(self):
        blocks = [TextBlock(text=["one"]), TextBlock(text=["two"])]
        project = FakeProject(blocks)
        translator = FakeTranslator()
        self.assertTrue(translate_project_textblocks(
            translator, project, "001.png", blocks[:1], full_page=False
        ))
        self.assertEqual(project.events, [])
        self.assertFalse(translator.calls[0][1]["full_page"])
```

- [ ] **Step 2: Add failing TranslateThread forwarding tests**

Construct a `TranslateThread` without starting it, attach a fake translator and project, invoke `_translate_page(project, "001.png", emit_finished=False)`, and assert the helper receives the exact project/page. Add a queue-order test where pages `001.png` then `002.png` are translated and `001.png` is marked before the second translator call snapshots context.

- [ ] **Step 3: Run pipeline tests to verify RED**

Run: `.venv/bin/python -m unittest tests.test_llm_context_pipeline -v`

Expected: FAIL because the helper and project-aware thread signatures do not exist.

- [ ] **Step 4: Implement one shared project translation helper**

Near `TranslateThread`, add:

```python
def translate_project_textblocks(
    translator,
    project,
    page_key,
    blocks,
    full_page=False,
):
    covers_page = translation_request_covers_full_page(
        blocks,
        project,
        page_key,
        full_page=full_page,
    )
    if covers_page:
        project.begin_full_page_translation(page_key)
    success = translator.translate_textblk_lst(
        blocks,
        project=project,
        page_key=page_key,
        full_page=covers_page,
    )
    if success and covers_page:
        project.mark_translation_finished(page_key, translator.lang_target)
    return bool(success)
```

Let exceptions propagate to the existing caller-specific error dialogs. This helper owns only completion ordering and context forwarding.

- [ ] **Step 5: Route standalone and parallel translation through the helper**

Change `TranslateThread._translate_page()` to accept `ProjImgTrans`, obtain `project.pages[page_key]`, call the helper with `full_page=True`, and return its boolean result. It must no longer swallow exceptions. Keep the standalone-page error dialog and `finish_translate_page` emission in the `translatePage()` job wrapper; let `_run_translate_pipeline()` retain its own existing exception dialog.

Change `TranslateThread.translatePage()` and `ModuleManager.translatePage()` to pass the project object rather than its pages dictionary. In `_run_translate_pipeline()`, assign `trans_success = self._translate_page(...)` so an `[ERROR: ...]` result is treated as failure even without an exception.

In `_run_translate_pipeline()`, call `_translate_page(self.imgtrans_proj, page_key, emit_finished=False)`. Remove the separate successful `update_page_progress(FIN_TRANSLATE)` because the helper now records both status and target atomically.

- [ ] **Step 6: Route direct, strict-order, and low-VRAM page translation through the helper**

Replace both direct `self.translator.translate_textblk_lst(blk_list)` calls in `ImgtransThread._imgtrans_pipeline()` with `translate_project_textblocks(..., full_page=True)`. Increment progress only after the helper returns; retain the existing model unload sequence. Do not add a second `FIN_TRANSLATE` update.

- [ ] **Step 7: Forward current project/page through selected-block translation**

Add keyword-only `page_key` to `ModuleManager.runBlktransPipeline()`, `_startBlktransPipeline()`, `ImgtransThread.runBlktransPipeline()`, and `_blktrans_pipeline()`. In `MainWindow.translateBlkitemList()`, pass `self.imgtrans_proj.current_img`.

In `_blktrans_pipeline()`, call `translate_project_textblocks()` with `full_page=False`; the shared coverage helper will commit only when every source-bearing page block is included. Keep OCR and inpaint stage behavior unchanged.

- [ ] **Step 8: Run pipeline, headless, relay, and API regressions**

Run: `.venv/bin/python -m unittest tests.test_llm_context_pipeline tests.test_headless_api tests.test_relay_jobs tests.test_relay_api tests.test_local_worker tests.test_api_uploads -v`

Expected: PASS; the pipeline supplies project context without changing headless/relay request contracts.

- [ ] **Step 9: Commit pipeline context forwarding**

```bash
git add ui/module_manager.py ui/mainwindow.py tests/test_llm_context_pipeline.py
git commit -m "feat: wire LLM context through translation pipelines"
```

---

### Task 9: User documentation and complete regression verification

**Files:**
- Modify: `doc/modules/translators.md`
- Modify: `tests/test_python_compat.py` only if the new package is not already covered by its repository-wide source discovery.

**Interfaces:**
- Documents the four settings, runtime-only history semantics, supported glossary formats, and failure behavior.
- Verifies the complete PR against current remote LLM, key-pool, prompt-preset, Gemma, pipeline, API, and Python 3.8 contracts.

- [ ] **Step 1: Add concrete remote and Gemma context documentation**

Under `LLM (Large Language Models)`, document:

````markdown
### Translation Context and Glossaries

Remote LLM translators and Gemma 4 expose the same optional context controls:

- `context mode=page` sends only the current page; `history` adds complete earlier translated pages.
- `history token budget` limits prior-page examples. Pages are removed whole, oldest first.
- `glossary path` accepts UTF-8 `.json`, `.txt`, and `.tsv` files.
- `glossary mode=matching` sends terms found on the current page; `all` sends every term.

JSON example:

```json
[
  {"src": "勇者", "dst": "용사", "info": "title"},
  {"src": "魔王", "dst": "마왕"}
]
```

TSV example:

```text
勇者<TAB>용사<TAB>title
魔王<TAB>마왕
```

TXT example:

```text
勇者->용사 # title
魔王->마왕
```

History is rebuilt from completed project pages at runtime and is not written as chat messages into the project file. Missing or invalid glossary files stop the request before the provider or Gemma worker runs.
````

Add the same four field names to both the remote-provider and local-Gemma settings lists without duplicating the full explanation.

- [ ] **Step 2: Run the focused PR 2 suite**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_translator_glossary \
  tests.test_llm_context_helpers \
  tests.test_llm_project_context \
  tests.test_llm_translation_context \
  tests.test_llm_context_pipeline \
  tests.test_llm_prompt_presets \
  tests.test_llm_api_key_pools \
  tests.test_llm_env \
  tests.test_local_translators \
  -v
```

Expected: PASS with no real network request, model download, or local Gemma runtime required.

- [ ] **Step 3: Run repository-wide unit tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 4: Verify Python 3.8 grammar and byte compilation**

Run:

```bash
.venv/bin/python -m unittest tests.test_python_compat -v
.venv/bin/python -m compileall -q modules utils ui tests
```

Expected: PASS with no syntax or import compilation failures.

- [ ] **Step 5: Inspect the final diff for accidental scope growth**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- modules/context modules/translators/base.py modules/translators/trans_llm_api_json.py modules/translators/trans_gemma4.py modules/translators/gemma4_worker.py utils/proj_imgtrans.py ui/module_manager.py ui/mainwindow.py tests doc/modules/translators.md
```

Expected:

- no whitespace errors;
- no changes to `config/textstyles/default.json` from this PR;
- no changes to API-key environment names, fixed provider names, model defaults, OCR/inpaint modules, or package namespace;
- only the files enumerated by this plan plus the user's pre-existing unrelated files appear.

- [ ] **Step 6: Commit documentation and final regression adjustments**

```bash
git add doc/modules/translators.md tests/test_python_compat.py
git commit -m "docs: document LLM translation context"
```

If `tests/test_python_compat.py` required no edit, omit it from `git add` and commit only the documentation.

---

## PR Acceptance Checklist

- [ ] Page-only mode with no glossary produces the same remote two-message request shape as before PR 2.
- [ ] All five fixed remote providers expose independent context parameter dictionaries.
- [ ] Gemma exposes the same context settings without changing its existing worker/runtime defaults.
- [ ] Matching glossary terms appear only in the current-page prompt; full glossary appears before history.
- [ ] History contains only complete, target-compatible, non-error earlier pages in chronological order.
- [ ] Rebuild, adjacent growth, bulk eviction, page jumps, project reloads, edited snapshots, and changed model/prompt/language/budget behave deterministically.
- [ ] Ordinary retries reuse immutable messages; only context recovery rebuilds with fewer whole pages.
- [ ] Failed requests do not mutate the committed window or mark project translation target metadata.
- [ ] Full-page, parallel, direct, low-VRAM, standalone, and selected-block flows pass project/page context correctly.
- [ ] Token logs expose aggregate counts only and never include user text, prompts, glossary contents, endpoints, or keys.
- [ ] Existing prompt preset, API-key pool, `.env`, Gemma chunk/retry/repair, headless, relay, local-worker, API, and Python 3.8 tests pass.
