# LLM Translation Context, History, Glossary, and Token Usage Design

## Goal

Add reusable cross-page translation context and glossary support to SekaiTranslator2's five OpenAI-compatible LLM translators and the local Gemma 4 translator without replacing Sekai's provider, API-key pool, prompt-preset, headless, relay, or local-worker behavior.

## Reference Inputs

- Product direction: `chatgpt-conversation://6a91a1db-83f4-83e8-9719-df72e7b51c34` (`비교 이식 요소 정리`), specifically PR 2: `context/history/glossary/token_usage` as a shared layer with Sekai-specific adapters.
- Reference implementation: [`SangGuKim/BallonsTranslator@0ca4965`](https://github.com/SangGuKim/BallonsTranslator/tree/0ca496533763e6945e8280989b174be1802fbded):
  - `ballontranslator/modules/context/history.py`
  - `ballontranslator/modules/context/glossary.py`
  - `ballontranslator/modules/context/token_usage.py`
  - `ballontranslator/modules/context/errors.py`
  - `ballontranslator/modules/translators/trans_llm.py`
- Both repositories use GPL-3.0, so adapted code remains under the repository's existing license.

## Chosen Approach

PR 2 is stacked after the custom-modules PR 1 baseline, but does not duplicate PR 1's discovery or namespace work.

Create a Qt-free `modules/context/` package and a thin `LLMContextAdapterMixin`. The core owns immutable history state, glossary parsing/selection, token estimation and usage formatting, and context-length error classification. Remote API and Gemma translators own their message shapes and provider/model behavior.

Expose context settings through the existing per-translator parameter schema instead of porting BallonsTranslator's global run-pipeline dialog. This matches Sekai's fixed-provider translator model: each provider already persists its own model, key pool, endpoint, and system prompt, while Gemma has a different practical token budget. It also supplies an existing editable selector and file-picker path for glossary files without creating a second settings UI.

The alternatives are intentionally rejected:

- Replacing `trans_llm_api_json.py` with BallonsTranslator's profile-backed `trans_llm.py` would discard Sekai's fixed-provider classes, API-key tiers, environment precedence, and prompt presets.
- Supporting only remote providers would leave the shared layer unavailable to Sekai's first-class local LLM path.
- Porting the global pipeline dialog would introduce unrelated UI and configuration migration work.

## User-Facing Configuration

Every fixed remote LLM translator and `Gemma 4 E4B-it` receives four settings:

| Key | Values | Default | Meaning |
|---|---|---|---|
| `context mode` | `page`, `history` | `page` | Translate the current page alone, or add eligible earlier pages as examples. |
| `history token budget` | positive integer | `4096` | Maximum estimated tokens reserved for prior page messages only. |
| `glossary path` | `.json`, `.txt`, `.tsv` file | empty | Empty disables glossary loading. |
| `glossary mode` | `matching`, `all` | `matching` | Include terms found on the current page, or the entire glossary. |

These values remain independent per fixed remote provider and per Gemma translator because they are stored in the existing `translator_params` entries. No global `ModuleConfig` fields or config-template migration are added.

When `context mode=page` and `glossary path` is empty, the remote request must preserve the existing two-message sequence and current user-prompt text. This is the compatibility default.

## Shared Context Core

### Glossary

The glossary loader is deterministic, UTF-8-only, and independent of Qt and provider SDKs.

Supported formats:

- JSON: an array of `{ "src": str, "dst": str, "info": optional str }` objects.
- TSV: `source<TAB>translation` with an optional third note column.
- TXT: either the TSV form or `source->translation #optional note`.
- Blank lines and lines beginning with `#`, `//`, or `\\` are ignored in text files.

Paths expand `~` and environment variables, normalize to an absolute real path, and cache by normalized path, nanosecond modification time, and file size. Exact duplicate rows collapse while file order is preserved. The same case-insensitive source mapped to different translations is an error with a precise file location.

`matching` mode performs case-folded literal matching against the joined current-page sources and returns each entry at most once in file order. `all` mode returns every entry. Rendering uses compact Unicode JSON so token estimates and request bytes remain stable.

### History

History is runtime state, not a second persisted copy of project text. The authoritative source is `ProjImgTrans.pages` plus page completion metadata.

An eligible history page must:

- precede the requested page in project order;
- have `FIN_TRANSLATE` set;
- have a matching `translation_target`, while accepting absent target metadata from legacy projects;
- contain at least one non-empty source;
- contain a non-empty translation for every non-empty source;
- contain no Sekai error marker beginning with `[ERROR:`.

Pages are immutable and indivisible. They are stored as source/translation snapshots plus provider-specific rendered user/assistant messages and a token cost. The runtime window is keyed by project load identity, source and target languages, model, system/style prompt, and history budget. Reopening the same path creates a new identity.

For a rebuild, select the newest chronological suffix that fits 60% of the configured budget, leaving headroom for adjacent growth. Adjacent successful pages append without rebuilding. When the hard history budget would overflow, evict whole oldest pages until the window is at or below the same 60% low-water mark, then append the previous page. Page jumps, project reloads, prompt/model/language/budget changes, changed retained snapshots, and incomplete preceding pages force a rebuild.

Only a fully successful page-level translation may advance the committed runtime window. Selected-block translation can use project context but advances the window only when the request covers every source-bearing block on that page. Ordinary retries reuse the same immutable request context.

### Token Usage and Context Errors

Remote message token counts use `tiktoken` when a model encoding is available and a deterministic fallback otherwise. The fallback counts ASCII runs as approximately four characters per token and non-ASCII characters as one token. Token usage logs normalize OpenAI-compatible `prompt`, `completion`, `total`, cache-hit, cache-miss, and cache-write fields without assuming one response schema.

Provider exceptions are classified as context-length failures only when their status/code/message clearly indicates oversized input. Authentication, rate-limit, timeout, and unrelated bad-request failures retain the current retry/key-rotation behavior.

On a remote context-length failure, evict at least one whole oldest history page toward the 60% low-water mark and retry without consuming the ordinary retry budget. Never truncate the current page or glossary. If no history remains, surface the failure through the translator's existing terminal error path.

Diagnostics contain page key, action, page count, estimated tokens, budget, appended count, evicted count, and rebuild reason. They never include source text, translations, glossary contents, prompts, endpoints, or API keys.

## Project and Translation Boundary

`ProjImgTrans` gains an opaque `load_identity`, `begin_full_page_translation(page_key)`, and `mark_translation_finished(page_key, target_language)`. Starting a full-page translation clears its previous translation completion and target metadata. Successful completion restores `FIN_TRANSLATE` and records the target language. Existing projects without target metadata continue to load.

`BaseTranslator.translate()` and `translate_textblk_lst()` accept keyword-only `project`, `page_key`, and completion intent while ordinary translators ignore them. A protected `_translate_with_context()` hook defaults to `_translate()` so existing translator subclasses do not change. LLM adapters override only that hook.

All full-page paths in `TranslateThread` and `ImgtransThread` pass the project and page key, invalidate completion before the request, and mark the target only after translation completes. Selected-block paths pass project/page context but do not mark the page complete unless all source-bearing blocks are translated.

## Remote LLM Adapter

The existing fixed-provider classes, API-key selection, endpoints, model catalog, Gemini sampling rules, response schema, Pydantic parsing, prompt presets, and error-marker convention remain in place.

The adapter changes request construction from a single prompt string to an immutable message list:

1. active `system_prompt`;
2. full glossary system constraint when `glossary mode=all`;
3. chronological history user/assistant page pairs;
4. current-page user prompt, with only matching glossary entries when `glossary mode=matching`.

History renderings are glossary-free. This prevents a glossary edit from contaminating stored history pairs and keeps the growing provider-cache prefix stable. The current JSON IDs and translation-count validation remain unchanged.

## Gemma Adapter

`LocalGGUFTranslator` uses the same context parameter schema and history snapshot/window rules. It serializes immutable history pages and the selected glossary into the existing worker payload; no model objects or mutable `TextBlock` instances cross the subprocess boundary.

The worker keeps current page chunking, strict JSON retry, suspicious-translation repair, and neighboring-cell context. For each current chunk, it constructs messages in the same order as the remote adapter and uses the loaded llama tokenizer to add only whole newest history pages that fit both `history token budget` and the chunk's remaining `max input tokens`. Current-page cells and the selected glossary are never removed to make room for history.

Because the Gemma model is loaded per subprocess call, its history window is a correctness and selection mechanism rather than a provider-cache optimization. The parent may retain the logical eligible window even when a particular chunk uses a smaller exact-fit suffix.

## Error Handling

- Missing, unreadable, unsupported, malformed, or conflicting glossaries fail before any remote request or Gemma subprocess starts.
- A failed translation does not commit a speculative history window or mark a page's target metadata.
- Ordinary remote retries keep identical messages. Context recovery is the only path allowed to rebuild them.
- Context overflow with no removable history follows the existing terminal error result instead of dropping current input.
- Gemma proactively trims whole history pages using its tokenizer; its current error strings and subprocess failure behavior remain intact.
- Disabling history or unloading/switching a translator clears its runtime history window.

## Testing

Tests cover:

1. glossary formats, path normalization/cache invalidation, matching/all selection, deterministic rendering, and precise errors;
2. token estimation fallback, usage formatting, and strict context-error classification;
3. rebuild/grow/evict/recovery history behavior and aggregate diagnostics;
4. project load identity and translation-target metadata lifecycle;
5. compatibility of ordinary translators with the new keyword-only boundary;
6. fixed-provider parameter isolation and unchanged page-only request shape;
7. remote matching/all glossary placement, immutable retries, whole-page overflow recovery, and commit-after-success;
8. Gemma payload serialization, exact-fit whole-page history, glossary placement, current chunk preservation, and existing retry/repair behavior;
9. full-page, parallel, low-VRAM, standalone page, and selected-block pipeline forwarding;
10. Python 3.8 grammar compatibility and the existing LLM/API-key/prompt/Gemma regression suites.

## Non-Goals

- Persisting conversation/history messages in project JSON.
- A graphical glossary editor, glossary import/export manager, or shared global glossary library.
- Speaker identification, geometry, image, OCR, inpaint, or multimodal context.
- Changing LLM OCR behavior.
- Replacing Sekai's provider classes, API-key pools, environment precedence, endpoints, model catalog, or system-prompt presets.
- Porting BallonsTranslator's LLM profile system, run-pipeline dialog, or package namespace.
- Adding text transforms, text-engine migration, or CTBD region filtering in this PR.
