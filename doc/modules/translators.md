# Ballon Translator: Translation Modules

*   Current codebase translators include Google, Papago, and OpenAI-compatible LLM providers.

[**Table of Contents**](#table-of-contents)
- [Ballon Translator: Translation Modules](#ballon-translator-translation-modules)
  - [LLM (Large Language Models)](#llm-large-language-models)
    - [Translation Context and Glossaries](#translation-context-and-glossaries)
    - [OpenAI-Compatible LLM Providers](#openai-compatible-llm-providers)
  - [Other Translators](#other-translators)
    - [Google](#google)
    - [Papago](#papago)
    - [Utility Translators](#utility-translators)
  - [Contributing to the Project](#contributing-to-the-project)

---

## LLM (Large Language Models)

The current LLM path uses JSON-structured responses. Each request asks the model to return a `translations` array with stable numeric IDs so the app can keep text blocks in order.

### Translation Context and Glossaries

The five fixed remote LLM translators listed below expose the same optional context controls. The settings are stored independently for each translator.

| Setting | Values | Default | Behavior |
|---|---|---|---|
| `context mode` | `page`, `history` | `page` | `page` sends only the current page; `history` adds complete earlier translated pages as examples. |
| `history token budget` | Positive integer | `4096` | Limits the estimated tokens reserved for prior-page examples. Pages are removed whole, oldest first, when the budget requires eviction. |
| `glossary path` | UTF-8 `.json`, `.txt`, or `.tsv` file | Empty | Selects a glossary file; an empty path disables glossary loading. |
| `glossary mode` | `matching`, `all` | `matching` | `matching` sends terms found on the current page; `all` sends every term. |

History is rebuilt from completed, target-compatible project pages at runtime and is not written as chat messages into the project file. Each history page remains an indivisible source/translation example: the complete page is either retained or evicted. Matching glossary entries are sent with the current-page prompt, while an `all` glossary is sent as a full constraint before history. Missing or invalid glossary files (including unreadable, unsupported, malformed, or conflicting files) stop the request before the provider request runs.

Glossary files must be UTF-8. Supported examples follow.

JSON:

```json
[
  {"src": "勇者", "dst": "용사", "info": "title"},
  {"src": "魔王", "dst": "마왕"}
]
```

TSV (`<TAB>` denotes a literal tab separator):

```text
勇者<TAB>용사<TAB>title
魔王<TAB>마왕
```

TXT:

```text
勇者->용사 # title
魔王->마왕
```

### OpenAI-Compatible LLM Providers

The following modules are provider-specific presets over the same OpenAI-compatible client:

*   **LLM OpenAI**
    *   Models shown in the UI: `OAI: gpt-5.2`, `OAI: gpt-5-mini`, `OAI: gpt-5-nano`
    *   Default endpoint: `https://api.openai.com/v1`
*   **LLM Google**
    *   Models shown in the UI: `GGL: gemini-3.6-flash`, `GGL: gemini-3.1-pro-preview`, `GGL: gemini-3.5-flash-lite`, `GGL: gemini-3-flash-preview`, `GGL: gemini-3.1-flash-lite`
    *   Default endpoint: `https://generativelanguage.googleapis.com/v1beta/openai`
    *   The same Gemini model and reasoning controls are available in `LLM OCR Google`.
    *   Requests for `gemini-3.6-flash` and `gemini-3.5-flash-lite` omit the legacy `temperature` and `top_p` fields. Older Gemini models retain the configured sampling values.
*   **LLM Grok**
    *   Models shown in the UI: `XAI: grok-4`, `XAI: grok-3`, `XAI: grok-3-mini`
    *   Default endpoint: `https://api.x.ai/v1`
*   **LLM OpenRouter**
    *   Uses the `override model` field.
    *   Default endpoint: `https://openrouter.ai/api/v1`
*   **LLM Studio**
    *   Uses the `override model` field.
    *   Requires a local endpoint such as `http://localhost:1234/v1`.

**Settings Fields:**

*   **apikey:** Single API key for the selected provider. If this is empty, the app reads `.env` or environment variables such as `BALLOONTRANS_LLM_OPENAI_API_KEY`, `BALLOONTRANS_LLM_GOOGLE_API_KEY`, `BALLOONTRANS_LLM_GROK_API_KEY`, or `BALLOONTRANS_LLM_OPENROUTER_API_KEY`.
*   **multiple_keys:** Semicolon-separated API keys. Requests rotate across keys and respect the per-key RPM limit. If this is empty, the app reads `BALLOONTRANS_LLM_<PROVIDER>_API_KEYS`.
*   **model:** Provider preset model.
*   **reasoning effort:** Gemini reasoning depth (`default`, `minimal`, `low`, `medium`, or `high`). `default` preserves the selected model's native behavior. Available in the Google translator and OCR presets.
*   **override model:** Custom model name. This is required for OpenRouter and LLM Studio unless the preset is enough for your endpoint.
*   **endpoint:** Base URL for the API. Leave blank for provider defaults, except for LLM Studio.
*   **system_prompt:** System message that defines the translator role and required JSON response.
*   **system prompt presets:** Provider-specific dropdown and editor for reusable system prompts. Use Add, Replace, and Delete to manage presets; `Default` cannot be deleted. Existing custom system prompts are retained as that provider's default.
*   **Japanese Doujin/Manga → Korean:** Built-in system-prompt preset for page-context-aware Japanese manga translation with natural Korean dialogue, stable character voice and speech levels, compact SFX handling, conservative OCR repair, and strict JSON output.
*   **invalid repeat count:** Retry count for response-count mismatches.
*   **max requests per minute:** RPM limit for each API key.
*   **delay:** Global delay in seconds between requests.
*   **max tokens:** Maximum response tokens.
*   **temperature / top_p:** Sampling controls. The latest Gemini request formats listed above omit these fields.
*   **retry attempts / retry timeout:** Retry behavior for transient API failures.
*   **proxy:** Optional proxy URL.
*   **context mode:** `page` (default) or `history`; see [Translation Context and Glossaries](#translation-context-and-glossaries).
*   **history token budget:** `4096` by default; see [Translation Context and Glossaries](#translation-context-and-glossaries).
*   **glossary path:** Empty by default; accepts UTF-8 `.json`, `.txt`, or `.tsv`; see [Translation Context and Glossaries](#translation-context-and-glossaries).
*   **glossary mode:** `matching` (default) or `all`; see [Translation Context and Glossaries](#translation-context-and-glossaries).

## Other Translators

### Google

**Attention:** Google Translate service has ceased operations in China.  If you are in China, you may need to use a VPN or proxy server to access Google Translate.

*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests. A value of `0.0` is usually sufficient.
*   **api_key:** Optional Google Translate API key. Empty uses `BALLOONTRANS_GOOGLE_TRANSLATE_API_KEY`.

### Papago

**Settings Fields:**

*   **delay:** Delay in seconds between requests to the translation service. Used to control the frequency of requests. A value of `0.0` is usually sufficient.

### Utility Translators

*   **None:** Leaves translation output empty.
*   **Copy Source:** Copies source text into the translation field.

---

## Contributing to the Project

*   To add a new translator, please refer to the [instructions](doc/how_to_add_new_translator.md). It's as simple as subclassing a BaseClass and implementing two interfaces. You are welcome to contribute to the project.
