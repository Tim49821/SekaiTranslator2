# Gemini reasoning effort and Gemini 3.5 Flash-Lite design

## Goal

Add Gemini reasoning-effort selection to both the Google LLM translator and Google LLM OCR, and make Gemini 3.5 Flash-Lite selectable in both modules.

## Scope

- Add `GGL: gemini-3.5-flash-lite` to the shared Google model catalog.
- Add a `reasoning effort` dropdown to `LLM Google` and `LLM OCR Google` only.
- Offer `default`, `minimal`, `low`, `medium`, and `high`.
- Preserve existing behavior when `default` is selected by omitting the API parameter.
- Send a non-default selection through the Gemini OpenAI-compatible API as `reasoning_effort`.
- Update the translator documentation and automated tests.

Other providers and unrelated UI components are outside this change.

## Architecture

The existing Google translator and OCR integrations share the model catalog defined by the LLM translator module. Gemini 3.5 Flash-Lite will be added there so both consumers receive the same model ID without duplicating it.

The generic parameter renderer already turns a parameter with `type: selector` into a dropdown. Each fixed Google module will add a provider-specific `reasoning effort` selector to its parameter definition. The selector will not appear in OpenAI, Grok, OpenRouter, LLM Studio, or Ollama modules.

The translator and OCR request builders will read the selected value. For Google requests, they will add `reasoning_effort` only when the selection is not `default`. This uses the field supported by Google's OpenAI-compatible Chat Completions endpoint and avoids changing the current OpenAI client integration.

## Data flow

1. The user selects a Google model and reasoning effort in the module settings.
2. The existing configuration system persists the selector value with the rest of the module parameters.
3. The Google request builder strips the `GGL:` model prefix as it does today.
4. If reasoning effort is `default`, the request omits `reasoning_effort` and Gemini applies the chosen model's native default.
5. Otherwise, the request includes `reasoning_effort` with `minimal`, `low`, `medium`, or `high`.

Google's compatibility layer maps `minimal` to the lowest supported level for models such as Gemini 3.1 Pro, where native `minimal` is unavailable.

## Error handling and compatibility

- Existing saved configurations receive `default` through the normal module-parameter patching path, preserving their current behavior.
- Only values present in the dropdown are emitted.
- Custom override model names continue to work. If the Google provider is selected, the chosen reasoning effort is still sent because support is determined by the configured provider, not by parsing the model name.
- API errors for a custom model that does not accept reasoning effort continue through the existing request error handling.

## Testing

Automated tests will verify:

- `GGL: gemini-3.5-flash-lite` appears in the shared translator and OCR Google model lists.
- Both fixed Google modules expose the same reasoning-effort options and default value.
- Non-Google fixed modules do not expose the Google-specific setting.
- Translator requests omit `reasoning_effort` for `default` and include the selected non-default value.
- OCR requests omit `reasoning_effort` for `default` and include the selected non-default value.
- Existing configuration migration and model-catalog tests continue to pass.

Implementation will follow test-driven development: each behavior will first be expressed by a failing test, followed by the smallest production change that makes it pass.
