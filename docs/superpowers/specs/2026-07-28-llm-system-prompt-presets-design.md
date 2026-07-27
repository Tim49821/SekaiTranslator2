# LLM System Prompt Presets Design

## Goal

Add a reusable system-prompt preset editor to each remote LLM translator. Presets must be stored independently for LLM OpenAI, LLM Google, LLM Grok, LLM OpenRouter, and LLM Studio. Ship one additional preset optimized for Japanese doujinshi and manga translation into natural Korean.

## Scope

This change covers the remote LLM translators implemented by `LLM_API_Translator`. It does not change the LLM OCR prompt UI or the local Gemma style-guide manager's saved data. It does not add cross-page memory or change the translation request payload beyond selecting the active system prompt.

## UI and Behavior

Each remote LLM translator will show a system-prompt preset manager in place of the standalone `system_prompt` editor. The manager contains:

- a preset dropdown;
- an inline editor containing the active prompt;
- `Add prompt`, `Replace prompt`, and `Delete prompt` actions.

Selecting a preset immediately copies its text into the active `system_prompt` parameter used by API requests. Editing the active prompt immediately updates the selected preset and active parameter, matching the current Gemma style-guide workflow. `Default` cannot be deleted, and the final remaining preset cannot be deleted.

The implementation will generalize the existing `ParamStyleGuideManager` into a parameter-driven preset manager. Its metadata specifies the active parameter key and UI labels, allowing the existing Gemma manager to retain its current appearance and behavior while the LLM system-prompt manager uses prompt-specific labels.

## Persistence and Compatibility

The new parameter is named `system prompt presets` and is saved inside the existing per-translator `translator_params` entry:

```json
{
  "selected": "Japanese Doujin/Manga → Korean",
  "styles": {
    "Default": "...",
    "Japanese Doujin/Manga → Korean": "..."
  }
}
```

Because every fixed-provider translator owns a deep copy of its parameter definitions and the configuration already saves parameters under the translator name, the five providers remain independent without a new global storage layer.

For existing configurations, the saved `system_prompt` value becomes that provider's `Default` preset during UI initialization. This preserves customized prompts instead of replacing them with the new built-in default. Newly created/default configurations receive both built-in presets.

The active `system_prompt` parameter remains present but hidden because the preset manager supplies its editor. API request code continues to read `self.system_prompt`, so selecting a preset requires no request-path changes.

## Built-in Manga Prompt

The preset `Japanese Doujin/Manga → Korean` will instruct the model to:

- translate Japanese into natural, concise Korean suitable for speech balloons;
- treat all supplied text cells as ordered, shared page context;
- preserve character voice, relationships, honorifics, speech levels, recurring terminology, emotional intensity, ambiguity, punctuation, and intentional repetition;
- handle SFX and mimetic language naturally without unnecessary expansion;
- repair only obvious OCR noise, including clearly misordered vertical Japanese, and avoid inventing missing content;
- avoid censorship, arbitrary softening, embellishment, summaries, and translator notes;
- return exactly one JSON translation for every requested numeric ID, with no markdown or extra prose.

The prompt is specifically named as Japanese-to-Korean so users do not accidentally assume it is target-language neutral. Language selection remains under user control; the preset does not automatically alter source or target selectors.

## Error Handling

- Empty or cancelled preset names do nothing.
- Adding an existing name replaces that named preset with the current editor content and selects it, preserving the existing manager behavior.
- Invalid saved selection names fall back to `Default`.
- Invalid or absent saved preset mappings fall back to the built-in presets.
- Deleting `Default` or the only remaining preset is ignored.

## Testing

Tests will verify:

1. every fixed remote LLM provider receives an independent deep-copied preset mapping;
2. the built-in manga prompt contains page-context, Korean-dialogue, OCR-repair, fidelity, and strict JSON requirements;
3. an existing customized `system_prompt` is adopted as the provider's `Default` preset;
4. selecting a preset emits and updates the active `system_prompt`;
5. adding, replacing, and deleting presets update the serialized manager state while protecting `Default`;
6. the existing Gemma style-guide manager behavior remains compatible.

## Non-Goals

- Sharing presets across providers.
- Importing or exporting prompt files.
- Automatically selecting the manga preset from the chosen languages.
- Sending images, geometry, speaker metadata, previous-page translations, or conversation history to the LLM.
- Changing response parsing or translation batching.
